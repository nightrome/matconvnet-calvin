function convertNetworkToFastRcnn2(obj, lastConvPoolName, finalFCLayerName)
% convertNetworkToFastRcnn(obj)
%
% Modify network for Fast R-CNN's ROI pooling
%
% Copyright by Holger Caesar, 2015
% Updated by Jasper Uijlings: Extra flexibility and possible bounding box regression
% Also added instanceWeights to loss layer

if nargin < 2 || isempty(lastConvPoolName)
    lastConvPoolName = 'pool5';
end

if nargin < 3 
    finalFCLayerName = [];
end

%% Add instanceWeights to loss layer. Note that this field remains empty when not
% given as input. So the loss layers should ignore empty instanceWeights.
softmaxInputs = obj.net.layers(obj.net.getLayerIndex('softmaxloss')).inputs;
if ~ismember('instanceWeights', softmaxInputs)
    softmaxInputs{end+1} = 'instanceWeights';
    obj.net.setLayerInputs('softmaxloss', softmaxInputs);
end


%% Replace pooling layer of last convolution layer with roiPooling
lastConvPoolIdx = obj.net.getLayerIndex(lastConvPoolName);
roiPoolName = ['roi' lastConvPoolName];
firstFCIdx = obj.net.layers(lastConvPoolIdx).outputIndexes;
assert(length(firstFCIdx) == 1);
firstFCName = obj.net.layers(lastConvPoolIdx).name;
roiPoolSize = obj.net.layers(firstFCIdx).block.size(1:2);
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
obj.net.replaceLayer(lastConvPoolName, roiPoolName, roiPoolBlock, {'oriImSize', 'boxes'}, {'roiPoolMask'});

%% Add bounding box regression layer
if ~isempty(finalFCLayerName)
    finalFCLayerIdx = obj.net.getLayerIndex(finalFCLayerName);
    inputVars = obj.net.layers(finalFCLayerIdx).inputs;
    finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);
    finalFCLayerSize = finalFCLayerSize .* [1 1 1 4]; % Four times bigger than classification layer
    regressName = [finalFCLayerName 'regress'];
    obj.net.addLayer(regressName, dagnn.Conv('size', finalFCLayerSize), inputVars, {'regressionScore'}, {'regressf', 'regressb'});
    fc8RegressIdx = obj.net.getLayerIndex(regressName);
    newParams = obj.net.layers(fc8RegressIdx).block.initParams();
    obj.net.params(obj.net.layers(fc8RegressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(fc8RegressIdx).paramIndexes(2)).value = newParams{2};
    
    obj.net.addLayer('regressLoss', dagnn.LossRegress('loss', 'Smooth', 'smoothMaxDiff', 1), ...
        {'regressionScore', 'regressionTargets', 'instanceWeights'}, 'regressObjective');
end

%% Set correct learning rates and biases (Girshick style)
% Biases have learning rate of 2 and no weight decay
for lI=1:length(obj.net.layers)
    if ~isempty(obj.net.layers(lI).paramIndexes)
        biasI = obj.net.layers(lI).paramIndexes(2);
        obj.net.params(biasI).learningRate = 2;
        obj.net.params(biasI).weightDecay = 0;
    end
end

% First convolutional layer does not learn
conv1I = obj.net.getLayerIndex('conv1');
obj.net.params(obj.net.layers(conv1I).paramIndexes(1)).learningRate = 0;
obj.net.params(obj.net.layers(conv1I).paramIndexes(2)).learningRate = 0;

%% If required, insert freeform pooling layer after roipool
if isfield(obj.nnOpts.misc, 'roiPool'),
    roiPool = obj.nnOpts.misc.roiPool;
    if isfield(roiPool, 'freeform') && roiPool.freeform.use
        % Compute activations for foreground and entire box separately
        % (by default off)
        roiPoolFreeformBlock = dagnn.RoiPoolingFreeform('combineFgBox', roiPool.freeform.combineFgBox);
        obj.net.insertLayer(roiPoolName, firstFCName, 'roipoolfreeform5', roiPoolFreeformBlock, 'blobMasks');
        
        % Share fully connected layer weights for foreground and entire box
        % (by default on)
        if isfield(roiPool.freeform, 'shareWeights') && ~roiPool.freeform.shareWeights
            relLayers = {'fc6', 'fc7', 'fc8'};
            for relIdx = 1 : numel(relLayers),
                relLayer = relLayers{relIdx};
                relLayerIdx = obj.net.getLayerIndex(relLayer);
                paramIndexes = obj.net.layers(relLayerIdx).paramIndexes;
                
                % Duplicate input size for all but the first fc layer
                if relIdx > 1
                    obj.net.layers(relLayerIdx).block.size(3) = obj.net.layers(relLayerIdx).block.size(3) * 2;
                    obj.net.params(paramIndexes(1)).value = cat(3, ...
                        obj.net.params(paramIndexes(1)).value, ...
                        obj.net.params(paramIndexes(1)).value);
                end
                % Duplicate output size for all but the last fc layers
                if relIdx < numel(relLayers)
                    obj.net.layers(relLayerIdx).block.size(4) = obj.net.layers(relLayerIdx).block.size(4) * 2;
                    obj.net.params(paramIndexes(1)).value = cat(4, ...
                        obj.net.params(paramIndexes(1)).value, ...
                        obj.net.params(paramIndexes(1)).value);
                    obj.net.params(paramIndexes(2)).value = cat(2, ...
                        obj.net.params(paramIndexes(2)).value, ...
                        obj.net.params(paramIndexes(2)).value);
                end
            end
        end
    end
end;