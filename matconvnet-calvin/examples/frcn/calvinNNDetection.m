% function calvinNNDetection()
%
% Copyright by Holger Caesar, 2016

% User inputs
global MYDATADIR % Directory of datasets
assert(~isempty(MYDATADIR));

%%% Settings
% Dataset
vocYear = 2010;
trainName = 'train';
testName  = 'val';
setupDataOpts(vocYear, testName);
global DATAopts; % Database specific paths
assert(~isempty(DATAopts), 'Error: Dataset not initialized properly!');
nnOpts.expDir = [DATAopts.resdir, 'Matconvnet-Calvin', '/', 'det', '/'];

% Detection-specific
nnOpts.testFn = @testDetection;
nnOpts.misc.overlapNms = 0.3;
nnOpts.derOutputs = {'objective', 1, 'regressObjective', 1};

% General
nnOpts.batchSize = 2;
nnOpts.numSubBatches = nnOpts.batchSize; % 1 image per sub-batch
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = [repmat(1e-3, 12, 1); repmat(1e-4, 4, 1)];
nnOpts.misc.netPath = fullfile(calvin_root(), 'data', 'Features', 'CNN-Models', 'matconvnet', 'imagenet-vgg-verydeep-16.mat');
nnOpts.gpus = SelectIdleGpu();

% Start from pretrained network
net = load(nnOpts.misc.netPath);

% Setup imdb
imdb = setupImdbDetection(trainName, testName, net);

% Create calvinNN CNN class
% By default, network is transformed into fast-rcnn with bbox regression
calvinn = CalvinNN(net, imdb, nnOpts);

% Train
calvinn.train();

% Test
stats = calvinn.test();

% Eval
evalDetection(testName, imdb, stats, nnOpts);