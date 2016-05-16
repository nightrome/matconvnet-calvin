function[stats] = fcnTestGeneric(varargin)
% [stats] = fcnTestGeneric(varargin)
%
% Copyright by Holger Caesar, 2016

% Initial settings
p = inputParser;
addParameter(p, 'dataset', SiftFlowDataset());
addParameter(p, 'modelType', 'fcn16s');
addParameter(p, 'gpus', 4);
addParameter(p, 'randSeed', 42);
addParameter(p, 'expNameAppend', 'test');
addParameter(p, 'epoch', 50);
addParameter(p, 'subset', 'test'); % train / test
addParameter(p, 'expSubFolder', '');
addParameter(p, 'findMapping', false); % Find the best possible label mapping from ILSVRC to target dataset
parse(p, varargin{:});

dataset = p.Results.dataset;
modelType = p.Results.modelType;
gpus = p.Results.gpus;
randSeed = p.Results.randSeed;
expNameAppend = p.Results.expNameAppend;
epoch = p.Results.epoch;
subset = p.Results.subset;
expSubFolder = p.Results.expSubFolder;
findMapping = p.Results.findMapping;
callArgs = p.Results; %#ok<NASGU>

% experiment and data paths
global glFeaturesFolder;
expName = [modelType, prependNotEmpty(expNameAppend, '-')];
expDir = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, expSubFolder, expName);
netPath = fullfile(expDir, sprintf('net-epoch-%d.mat', epoch));
settingsPath = fullfile(expDir, 'settings.mat');

% Fix randomness
rng(randSeed);

% Check if directory exists
if ~exist(expDir, 'dir')
    error('Error: Experiment directory does not exist %s\n', expDir);
end;

% Load imdb and nnOpts from file
if exist(settingsPath, 'file')
    settingsStruct = load(settingsPath, 'callArgs', 'nnOpts', 'imdbFcn');
    nnOpts = settingsStruct.nnOpts;
    imdbFcn = settingsStruct.imdbFcn;
    assert(isequal(dataset.name, imdbFcn.dataset.name));
else
    error('Error: Cannot find settings path: %s', settingsPath);
end

%%% Compatibility stuff
if ~iscell(imdbFcn.data.train)
    imdbFcn.data.train = imdbFcn.imdb.images.name(imdbFcn.data.train);
    imdbFcn.data.val   = imdbFcn.imdb.images.name(imdbFcn.data.val);
end

% Workaround set val as test
if ~isfield(imdbFcn.data, 'test')
    imdbFcn.data.test = imdbFcn.data.val;
    imdbFcn.data = rmfield(imdbFcn.data, 'val');
end

imdbFcn.numClasses = imdbFcn.dataset.labelCount;
%%%

% Overwrite some settings
nnOpts.gpus = gpus;
nnOpts.convertToTrain = false;
nnOpts.expDir = expDir;

% Create network
nnClass = FCNNN(netPath, imdbFcn, nnOpts);

% Test the network
stats = nnClass.testOnSet('subset', subset, 'findMapping', findMapping);