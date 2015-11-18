function convertNetwork(obj, net)
% convertNetwork(obj, net)
%
% Converts a Matconvnet network into the equivalent Fast R-CNN network,
% expressed as a Directed Acyclic Graph.
%
% Copyright by Holger Caesar, 2015

% Use the default routine to convert an image class. network to FRCN
net = dagnn.DagNN.fromSimpleNN(net);

% Remove unused/incorrect fields from normalization
net.meta.normalization = rmfield(net.meta.normalization, 'keepAspect');
net.meta.normalization = rmfield(net.meta.normalization, 'border');
net.meta.normalization = rmfield(net.meta.normalization, 'imageSize');
net.meta.normalization = rmfield(net.meta.normalization, 'interpolation');

% Add dropout layers after relu6 and relu7
dropout6Layer = dagnn.DropOut();
dropout7Layer = dagnn.DropOut();
net.insertLayer('relu6', 'fc7', 'dropout6', dropout6Layer);
net.insertLayer('relu7', 'fc8', 'dropout7', dropout7Layer);

% Replace softmax with softmaxloss for training
softmaxlossBlock = dagnn.Loss('loss', 'softmaxlog');
net.replaceLayer('prob', 'softmaxloss', softmaxlossBlock, 'label');

% Adapt number of classes in softmaxloss layer from 1000 to numClasses
fc8Idx = net.getLayerIndex('fc8');
net.layers(fc8Idx).block.size(4) = obj.imdb.numClasses;
newParams = net.layers(fc8Idx).block.initParams();
net.params(net.layers(fc8Idx).paramIndexes(1)).value = newParams{1};
net.params(net.layers(fc8Idx).paramIndexes(2)).value = newParams{2};

% Modify network for Fast R-CNN's ROI pooling
if isfield(obj.nnOpts, 'roiPool') && obj.nnOpts.roiPool.use,
    % Replace max-pooling layer by ROI pooling
    fc6Idx = net.getLayerIndex('fc6');
    roiPoolSize = net.layers(fc6Idx).block.size(1:2);
    roiPoolBlock = dagnn.ROIPooling('poolSize', roiPoolSize);
    net.replaceLayer('pool5', 'roipool5', roiPoolBlock, {'oriImSize', 'boxes'}, {'roiPool5Mask'});
    
    % If required, insert freeform pooling layer after roipool
    if isfield(obj.nnOpts.roiPool, 'freeform') && obj.nnOpts.roiPool.freeform.use,
        roiPoolFreeformBlock = dagnn.ROIPoolingFreeform('combineFgBox', true);
        net.insertLayer('roipool5', 'fc6', 'roipoolfreeform5', roiPoolFreeformBlock, 'blobMasks');
    end;
end;

% Rename input and output
net.renameVar('x0', 'input');
net.renameVar(net.layers(net.getLayerIndex('softmaxloss')).outputs, 'objective');

% Update class fields
obj.net = net;