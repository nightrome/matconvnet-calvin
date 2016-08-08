% function calvinNNClassification

% User inputs
global MYDATADIR % Directory of datasets
assert(~isempty(MYDATADIR));

%%% Settings
% Dataset
vocYear = 2010;
trainName = 'train';
testName  = 'val';

% Task
nnOpts.lossFnObjective = 'hinge';
nnOpts.testFn = @testClassification;
nnOpts.derOutputs = {'objective', single(1)};
targetImSize = [224, 224];

% Disable Fast R-CNN (default is on)
nnOpts.fastRcnn = false;
nnOpts.fastRcnnParams = false; % learning rates and weight decay
nnOpts.misc.roiPool.use = false;
nnOpts.misc.roiPool.freeform.use = false;
nnOpts.bboxRegress = false;

% General
nnOpts.batchSize = 64;
nnOpts.numSubBatches = 1;  % 64 images per sub-batch
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = [repmat(1e-3, 12, 1); repmat(1e-4, 4, 1)];
nnOpts.misc.netPath = fullfile(MYDATADIR, '..', 'MatconvnetModels', 'imagenet-vgg-verydeep-16.mat');
nnOpts.gpus = SelectIdleGpu();

% Setup data opts
setupDataOpts(vocYear, testName);
global DATAopts; % Database specific paths
nnOpts.expDir = [DATAopts.resdir, 'Matconvnet-Calvin', '/', 'cls', '/'];

% Start from pretrained network
net = load(nnOpts.misc.netPath);

% Setup imdb
imdb = setupImdbClassification(trainName, testName, net);
imdb.targetImSize = targetImSize;

% Create calvinNN CNN class
calvinn = CalvinNN(net, imdb, nnOpts);

% Train
calvinn.train();

% Test
stats = calvinn.test();

% Eval
evalClassification(imdb, stats, nnOpts);