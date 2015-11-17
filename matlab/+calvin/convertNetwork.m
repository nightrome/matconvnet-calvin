function[net] = convertNetwork(net, imdb, nnOpts)
% [net] = convertNetwork(net, imdb, nnOpts)
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
insertLayer(net, 'relu6', 'fc7', 'dropout6', dropout6Layer);
insertLayer(net, 'relu7', 'fc8', 'dropout7', dropout7Layer);

% Replace softmax with softmaxloss for training
softmaxlossLayer = dagnn.Loss('loss', 'softmaxlog');
replaceLayer(net, 'prob', 'softmaxloss', softmaxlossLayer, 'label');

% Adapt number of classes in softmaxloss layer from 1000 to labelCount
fc8Idx = net.getLayerIndex('fc8');
net.layers(fc8Idx).block.size(4) = imdb.labelCount;
newParams = net.layers(fc8Idx).block.initParams();
net.params(net.layers(fc8Idx).paramIndexes(1)).value = newParams{1};
net.params(net.layers(fc8Idx).paramIndexes(2)).value = newParams{2};

% Modify network for Fast R-CNN's ROI pooling
if isfield(nnOpts, 'roiPool'),
    % Replace max-pooling layer by ROI pooling
    fc6Idx = net.getLayerIndex('fc6');
    roiPoolSize = net.layers(fc6Idx).block.size(1:2);
    roiPoolLayer = ROIPooling('poolSize', roiPoolSize);
    replaceLayer(net, 'pool5', 'roiPool5', roiPoolLayer, {'oriImSize', 'boxes'}, {'roiPool5Mask'});
    
    % If required, insert freeform pooling layer after roipool
    if isfield(nnOpts.roiPool, 'roiPoolFreeform') && nnOpts.roiPool.roiPoolFreeform,
        roiPoolFreeformLayer = ROIPoolingFreeform();
        insertLayer(net, 'roiPool5', 'fc6', 'roiPoolFreeform5', roiPoolFreeformLayer, 'batchAux');
    end;
end;

% Rename input and output
net.renameVar('x0', 'input');
net.renameVar(net.layers(net.getLayerIndex('softmaxloss')).outputs, 'objective');

function[net] = replaceLayer(net, layerName, newLayerName, newLayer, addInputs, addOutputs, addParams)
% [net] = replaceLayer(net, layerName, newLayerName, newLayer, [addInputs], [addOutputs], [addParams])
%
% Takes a DAG and replaces an existing layer with a new one.
% If not specified, the inputs, outputs and params are reused from the
% previous layer.
%
% Copyright by Holger Caesar, 2015

% Find the old layer
layerIdx = net.getLayerIndex(layerName);
layer = net.layers(layerIdx);

newInputs  = layer.inputs;
newOutputs = layer.outputs;
newParams  = layer.params;

if exist('addInputs', 'var') && ~isempty(addInputs),
    newInputs = [newInputs, addInputs];
end;
if exist('addOutputs', 'var') && ~isempty(addOutputs),
    newOutputs = [newOutputs, addOutputs];
end;
if exist('addParams', 'var') && ~isempty(addParams),
    newParams = [newParams, addParams];
end;

% Add the new layer
net.addLayer(newLayerName, newLayer, newInputs, newOutputs, newParams);

% Remove the old layer
net.removeLayer(layerName);

function [net] = insertLayer(net, leftLayerName, rightLayerName, newLayerName, newLayer, addInputs, addOutputs, newParams)
% [net] = insertLayer(net, leftLayerName, rightLayerName, newLayerName, newLayer, [addInputs], [addOutputs], [newParams])
%
% Takes a DAG and inserts a new layer before an existing layer.
% The outputs of the previous layer and inputs of the following layer are
% adapted accodingly.
%
% Copyright by Holger Caesar, 2015

% Find the old layers and their outputs/inputs
leftLayerIdx = net.getLayerIndex(leftLayerName);
rightLayerIdx = net.getLayerIndex(rightLayerName);
leftLayer = net.layers(leftLayerIdx);
rightLayer = net.layers(rightLayerIdx);
leftOutputs  = leftLayer.outputs;
rightInputs = rightLayer.inputs;

% Check whether left and right are actually connected
assert(leftLayerIdx ~= rightLayerIdx);

% Introduce new free variables for new layer outputs
rightInputs = replaceVariables(net, rightInputs);

% Change the input of the right layer (to avoid cycles)
net.layers(rightLayerIdx).inputs = rightInputs;

newInputs = leftOutputs;

% Remove special inputs from rightInputs (i.e. labels)
newOutputs = regexp(rightInputs, '^(x\d+)', 'match', 'once');
newOutputs = newOutputs(~cellfun(@isempty, newOutputs));

% Adapt inputs, outputs and params
if exist('addInputs', 'var') && ~isempty(addInputs),
    newInputs = [newInputs, addInputs];
end;
if exist('addOutputs', 'var') && ~isempty(addOutputs),
    newOutputs = [newOutputs, addOutputs];
end;
if ~exist('newParams', 'var') || isempty(newParams),
    newParams = {};
end;

% Add the new layer
net.addLayer(newLayerName, newLayer, newInputs, newOutputs, newParams);

function[variables] = replaceVariables(net, variables)
% [variables] = replaceVariables(net, variables)
%
% Replace all default variables (x\d+) with free variables.
%
% Copyright by Holger Caesar, 2015

for i = 1 : numel(variables),
    oldVariable = variables{i};
    
    if regexp(oldVariable, 'x\d+'),
        freeVariable = getFreeVariable(net);
        variables{i} = freeVariable;
    end;
end;

function[var] = getFreeVariable(net)
% [var] = getFreeVariable(net)
%
% Returns the name of a new free variable.
% The name is xi where i is the smallest unused integer in the existing
% variables.
%
% Copyright by Holger Caesar, 2015

names = {net.vars.name};
inds = nan(numel(names, 1));

for i = 1 : numel(names),
    [tok, ~] = regexp(names{i}, 'x(\d+)', 'tokens', 'match');
    if ~isempty(tok),
        inds(i) = str2double(tok{1});
    end;
end;

maxInd = max(inds);
var = sprintf('x%d', maxInd+1);