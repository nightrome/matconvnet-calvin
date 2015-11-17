function insertLayer(obj, leftLayerName, rightLayerName, newLayerName, newLayer, addInputs, addOutputs, newParams)
% insertLayer(obj, leftLayerName, rightLayerName, newLayerName, newLayer, [addInputs], [addOutputs], [newParams])
%
% Takes a DAG and inserts a new layer before an existing layer.
% The outputs of the previous layer and inputs of the following layer are
% adapted accodingly.
%
% Copyright by Holger Caesar, 2015

% Find the old layers and their outputs/inputs
leftLayerIdx = obj.getLayerIndex(leftLayerName);
rightLayerIdx = obj.getLayerIndex(rightLayerName);
leftLayer = obj.layers(leftLayerIdx);
rightLayer = obj.layers(rightLayerIdx);
leftOutputs  = leftLayer.outputs;
rightInputs = rightLayer.inputs;

% Check whether left and right are actually connected
assert(leftLayerIdx ~= rightLayerIdx);

% Introduce new free variables for new layer outputs
rightInputs = replaceVariables(obj, rightInputs);

% Change the input of the right layer (to avoid cycles)
obj.layers(rightLayerIdx).inputs = rightInputs;

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
obj.addLayer(newLayerName, newLayer, newInputs, newOutputs, newParams);

function[variables] = replaceVariables(obj, variables)
% [variables] = replaceVariables(obj, variables)
%
% Replace all default variables (x\d+) with free variables.
%
% Copyright by Holger Caesar, 2015

for i = 1 : numel(variables),
    oldVariable = variables{i};
    
    if regexp(oldVariable, 'x\d+'),
        freeVariable = obj.getFreeVariable();
        variables{i} = freeVariable;
    end;
end;

function[var] = getFreeVariable(obj)
% [var] = getFreeVariable(obj)
%
% Returns the name of a new free variable.
% The name is xi where i is the smallest unused integer in the existing
% variables.
%
% Copyright by Holger Caesar, 2015

names = {obj.vars.name};
inds = nan(numel(names, 1));

for i = 1 : numel(names),
    [tok, ~] = regexp(names{i}, 'x(\d+)', 'tokens', 'match');
    if ~isempty(tok),
        inds(i) = str2double(tok{1});
    end;
end;

maxInd = max(inds);
var = sprintf('x%d', maxInd+1);