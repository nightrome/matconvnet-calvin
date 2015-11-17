function replaceLayer(obj, layerName, newLayerName, newLayer, addInputs, addOutputs, addParams)
% replaceLayer(obj, layerName, newLayerName, newLayer, [addInputs], [addOutputs], [addParams])
%
% Takes a DAG and replaces an existing layer with a new one.
% If not specified, the inputs, outputs and params are reused from the
% previous layer.
%
% Copyright by Holger Caesar, 2015

% Find the old layer
layerIdx = obj.getLayerIndex(layerName);
layer = obj.layers(layerIdx);

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
obj.addLayer(newLayerName, newLayer, newInputs, newOutputs, newParams);

% Remove the old layer
obj.removeLayer(layerName);