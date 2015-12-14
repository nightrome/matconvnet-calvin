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
if isfield(net.meta, 'normalization');
    net.meta.normalization = rmfield(net.meta.normalization, 'keepAspect');
    net.meta.normalization = rmfield(net.meta.normalization, 'border');
    net.meta.normalization = rmfield(net.meta.normalization, 'imageSize');
    net.meta.normalization = rmfield(net.meta.normalization, 'interpolation');
end

% Add dropout layers after relu6 and relu7
dropout6Layer = dagnn.DropOut();
dropout7Layer = dagnn.DropOut();
net.insertLayer('relu6', 'fc7', 'dropout6', dropout6Layer);
net.insertLayer('relu7', 'fc8', 'dropout7', dropout7Layer);

% Replace softmax with softmaxloss for training
softmaxlossBlock = dagnn.LossWeighted('loss', 'softmaxlog');
net.replaceLayer('prob', 'softmaxloss', softmaxlossBlock, 'label');

% Adapt number of classes in softmaxloss layer from 1000 to numClasses
fc8Idx = net.getLayerIndex('fc8');
net.layers(fc8Idx).block.size(4) = obj.imdb.numClasses;
newParams = net.layers(fc8Idx).block.initParams();
net.params(net.layers(fc8Idx).paramIndexes(1)).value = newParams{1} / std(newParams{1}(:)) * 0.01; % Girshick initialization
net.params(net.layers(fc8Idx).paramIndexes(2)).value = newParams{2}';

% Rename input and output
net.renameVar('x0', 'input');
net.renameVar(net.layers(net.getLayerIndex('softmaxloss')).outputs, 'objective');

% Update class fields
obj.net = net;