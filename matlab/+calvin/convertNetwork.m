function[obj] = convertNetwork(net, imdb, nnOpts)
% [obj] = convertNetwork(net, imdb, nnOpts)
%
% Converts a Matconvnet network into the equivalent Fast R-CNN network,
% expressed as a Directed Acyclic Graph.
%
% Copyright by Holger Caesar, 2015

% Use the default routine to convert an image class. network to FRCN
obj = dagnn.DagNN.fromSimpleNN(net);

% Remove unused fields from normalization
obj.meta.normalization = rmfield(obj.meta.normalization, 'keepAspect');
obj.meta.normalization = rmfield(obj.meta.normalization, 'border');
obj.meta.normalization = rmfield(obj.meta.normalization, 'imageSize');
obj.meta.normalization = rmfield(obj.meta.normalization, 'interpolation');

% Add dropout layers after relu6 and relu7
dropout6Layer = dagnn.DropOut();
dropout7Layer = dagnn.DropOut();
obj = DagNN_insertLayer(obj, 'relu6', 'fc7', 'dropout6', dropout6Layer);
obj = DagNN_insertLayer(obj, 'relu7', 'fc8', 'dropout7', dropout7Layer);

% Replace softmax with softmaxloss for training
softmaxlossLayer = dagnn.Loss('loss', 'softmaxlog');
obj = DagNN_replaceLayer(obj, 'prob', 'softmaxloss', softmaxlossLayer, 'label');

% Adapt number of classes in softmaxloss layer from 1000 to labelCount
fc8Idx = obj.getLayerIndex('fc8');
obj.layers(fc8Idx).block.size(4) = imdb.labelCount;
newParams = obj.layers(fc8Idx).block.initParams();
obj.params(obj.layers(fc8Idx).paramIndexes(1)).value = newParams{1};
obj.params(obj.layers(fc8Idx).paramIndexes(2)).value = newParams{2};

% Modify network for Fast R-CNN's ROI pooling
if isfield(nnOpts, 'roiPool'),
    % Replace max-pooling layer by ROI pooling
    fc6Idx = obj.getLayerIndex('fc6');
    roiPoolSize = obj.layers(fc6Idx).block.size(1:2);
    roiPoolLayer = ROIPooling('poolSize', roiPoolSize);
    obj = DagNN_replaceLayer(obj, 'pool5', 'roiPool5', roiPoolLayer, {'oriImSize', 'boxes'}, {'roiPool5Mask'});
    
    % If required, insert freeform pooling layer after roipool
    if isfield(nnOpts.roiPool, 'roiPoolFreeform') && nnOpts.roiPool.roiPoolFreeform,
        roiPoolFreeformLayer = ROIPoolingFreeform();
        obj = DagNN_insertLayer(obj, 'roiPool5', 'fc6', 'roiPoolFreeform5', roiPoolFreeformLayer, 'batchAux');
    end;
end;

% Rename input and output
obj.renameVar('x0', 'input');
obj.renameVar(obj.layers(obj.getLayerIndex('softmaxloss')).outputs, 'objective');