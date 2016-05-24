function convertNetwork(obj)
% convertNetwork(obj)
%
% Converts a network from test to train, inserting lossed, dropout and
% adopting the classification layer.
%
% Copyright by Holger Caesar, 2015

fprintf('Converting test network to train network (dropout, loss, etc.)...\n');

% Add dropout or batch normalization
if isfield(obj.nnOpts.misc, 'batchNorm') && obj.nnOpts.misc.batchNorm
    % Add batch normalization after all fc and conv layers
    
    fprintf('Adding batch normalization to network...\n');
    
    % Get ReLUs
    reluInds = find(arrayfun(@(x) isa(x.block, 'dagnn.ReLU'), obj.net.layers));
    
    for i = 1 : numel(reluInds)
        % Relu
        reluIdx = reluInds(i);
        reluLayerName = obj.net.layers(reluIdx).name;
        reluInputIdx = obj.net.layers(reluIdx).inputIndexes;
        assert(numel(reluInputIdx) == 1);
        
        % Left layer
        leftLayerIdx = find(arrayfun(@(x) ismember(reluInputIdx, x.outputIndexes), obj.net.layers));
        assert(numel(leftLayerIdx) == 1);
        leftLayerName = obj.net.layers(leftLayerIdx).name;
        leftParamIdx = obj.net.layers(leftLayerIdx).paramIndexes(1);
        numChannels = size(obj.net.params(leftParamIdx).value, 4); % equals size(var, 3) of the input variable
        
        % Insert new layer
        layerBlock = dagnn.BatchNorm('numChannels', numChannels);
        layerParamValues = layerBlock.initParams();
        layerName = sprintf('bn_%s', reluLayerName);
        layerParamNames = cell(1, numel(layerParamValues));
        for i = 1 : numel(layerParamValues) %#ok<FXSET>
            layerParamNames{i} = sprintf('%s_%d', layerName, i);
        end
        insertLayer(obj.net, leftLayerName, reluLayerName, layerName, layerBlock, {}, {}, layerParamNames);
        
        for i = 1 : numel(layerParamValues) %#ok<FXSET>
            paramIdx = obj.net.getParamIndex(layerParamNames{i});
            obj.net.params(paramIdx).value = layerParamValues{i};
            
            % Learning rate and weight decay as taken from Matconvnet's cnn_imagenet_init.m
            if i == 1
                obj.net.params(paramIdx).learningRate = 2;
            elseif i == 2
                obj.net.params(paramIdx).learningRate = 1;
            elseif i == 3
                obj.net.params(paramIdx).learningRate = 0.05;
            else
                error('Error: Invalid number of batch norm parameters!');
            end
            obj.net.params(paramIdx).weightDecay = 0;
        end
    end
else
    % Add dropout layers after relu6 and relu7
    
    fprintf('Adding dropout to network...\n');
    
    dropout6Layer = dagnn.DropOut();
    dropout7Layer = dagnn.DropOut();
    insertLayer(obj.net, 'relu6', 'fc7', 'dropout6', dropout6Layer);
    insertLayer(obj.net, 'relu7', 'fc8', 'dropout7', dropout7Layer);
end

% Rename variable prob to x%d+ style to make insertLayer work
% (Required for beta18 or later matconvnet default networks)
predVarIdx = obj.net.getVarIndex('prediction');
if ~isnan(predVarIdx)
    freeVarName = getFreeVariable(obj.net);
    obj.net.renameVar('prediction', freeVarName);
end

% Replace softmax with correct loss for training (default: softmax)
switch obj.nnOpts.lossFnObjective
    case 'softmaxlog'
        softmaxlossBlock = dagnn.LossWeighted('loss', 'softmaxlog');
        replaceLayer(obj.net, 'prob', 'softmaxloss', softmaxlossBlock, 'label');
        objectiveName = obj.net.layers(obj.net.getLayerIndex('softmaxloss')).outputs;
        obj.net.renameVar(objectiveName, 'objective');
    case 'hinge'
        hingeLossBlock = dagnn.Loss('loss', 'hinge');
        replaceLayer(obj.net, 'prob', 'hingeloss', hingeLossBlock, 'label');
        objectiveName = obj.net.layers(obj.net.getLayerIndex('hingeloss')).outputs;
        obj.net.renameVar(objectiveName, 'objective');
    otherwise
        error('Wrong loss specified');
end

% Adapt number of classes in softmaxloss layer from 1000 to numClasses
fc8Idx = obj.net.getLayerIndex('fc8');
obj.net.layers(fc8Idx).block.size(4) = obj.imdb.numClasses;
newParams = obj.net.layers(fc8Idx).block.initParams();
obj.net.params(obj.net.layers(fc8Idx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.01; % Girshick initialization
obj.net.params(obj.net.layers(fc8Idx).paramIndexes(2)).value = newParams{2}';

% Rename input
if ~isnan(obj.net.getVarIndex('x0'))
    % Input variable x0 is renamed to input
    obj.net.renameVar('x0', 'input');
else
    % Input variable already has the correct name
    assert(~isnan(obj.net.getVarIndex('input')));
end

% Modify for Fast Rcnn (ROI pooling, bbox regression etc.)
if obj.nnOpts.fastRcnn
    obj.convertNetworkToFastRcnn();
end