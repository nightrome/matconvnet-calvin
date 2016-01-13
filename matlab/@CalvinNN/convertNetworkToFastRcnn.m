function convertNetworkToFastRcnn(obj, varargin)
% convertNetworkToFastRcnn(obj, varargin)
%
% Modify network for Fast R-CNN's ROI pooling.
%
% Copyright by Holger Caesar, 2015
% Updated by Jasper Uijlings:
%  - Extra flexibility and possible bounding box regression
%  - Added instanceWeights to loss layer

% Initial settings
p = inputParser;
addParameter(p, 'lastConvPoolName', 'pool5');
addParameter(p, 'firstFCName', 'fc6');
addParameter(p, 'finalFCName', 'fc8');
parse(p, varargin{:});

lastConvPoolName = p.Results.lastConvPoolName;
firstFCName = p.Results.firstFCName;
finalFCName = p.Results.finalFCName;

%%% Add weights to loss layer. Note that this field remains empty when not
% given as input. So the loss layers should ignore empty weights.
softmaxInputs = obj.net.layers(obj.net.getLayerIndex('softmaxloss')).inputs;
if ~ismember('instanceWeights', softmaxInputs)
    softmaxInputs{end+1} = 'instanceWeights';
    obj.net.setLayerInputs('softmaxloss', softmaxInputs);
end

%%% Replace pooling layer of last convolution layer with roiPooling
lastConvPoolIdx = obj.net.getLayerIndex(lastConvPoolName);
assert(~isnan(lastConvPoolIdx));
roiPoolName = ['roi', lastConvPoolName];
firstFCIdx = obj.net.layers(lastConvPoolIdx).outputIndexes;
assert(length(firstFCIdx) == 1);
roiPoolSize = obj.net.layers(firstFCIdx).block.size(1:2);
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
obj.net.replaceLayer(lastConvPoolName, roiPoolName, roiPoolBlock, {'oriImSize', 'boxes'}, {'roiPoolMask'});

%%% Add bounding box regression layer
if obj.nnOpts.bboxRegress
    finalFCLayerIdx = obj.net.getLayerIndex(finalFCName);
    inputVars = obj.net.layers(finalFCLayerIdx).inputs;
    finalFCLayerSize = size(obj.net.params(obj.net.layers(finalFCLayerIdx).paramIndexes(1)).value);
    regressLayerSize = finalFCLayerSize .* [1 1 1 4]; % Four times bigger than classification layer
    regressName = [finalFCName 'regress'];
    obj.net.addLayer(regressName, dagnn.Conv('size', regressLayerSize), inputVars, {'regressionScore'}, {'regressf', 'regressb'});
    regressIdx = obj.net.getLayerIndex(regressName);
    newParams = obj.net.layers(regressIdx).block.initParams();
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.001; % Girshick initialization with std of 0.001
    obj.net.params(obj.net.layers(regressIdx).paramIndexes(2)).value = newParams{2};

    obj.net.addLayer('regressLoss', dagnn.LossRegress('loss', 'Smooth', 'smoothMaxDiff', 1), ...
        {'regressionScore', 'regressionTargets', 'instanceWeights'}, 'regressObjective');
end

%%% Set correct learning rates and biases (Girshick style)
% Biases have learning rate of 2 and no weight decay
for lI=1:length(obj.net.layers)
    if isa(obj.net.layers(lI).block, 'dagnn.Conv')
        biasI = obj.net.layers(lI).paramIndexes(2);
        obj.net.params(biasI).learningRate = 2;
        obj.net.params(biasI).weightDecay = 0;
    end
end

% First convolutional layer should not learn
% Note that this is different from Fast R-CNN, but gives good results
% Also note that there is no speedup as Matconvnet still computes the
% gradients, but does not update the parameters.
conv1I = obj.net.getLayerIndex('conv1'); % AlexNet-style networks
if isnan(conv1I)
    conv1I = obj.net.getLayerIndex('conv1_1'); % VGG-16 style networks
end
obj.net.params(obj.net.layers(conv1I).paramIndexes(1)).learningRate = 0;
obj.net.params(obj.net.layers(conv1I).paramIndexes(2)).learningRate = 0;

%%% If required, insert freeform pooling layer after roipool
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
            fcLayers = {obj.net.layers(~cellfun(@isempty, regexp({obj.net.layers.name}, 'fc.*'))).name};
            for relIdx = 1 : numel(fcLayers)
                relLayer = fcLayers{relIdx};
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
                if relIdx < numel(fcLayers)
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
end