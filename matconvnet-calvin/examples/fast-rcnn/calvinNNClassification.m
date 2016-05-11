% function calvinNNClassification

% User inputs
global MYDATADIR        % Directory of datasets

%%% Settings
% Dataset
vocYear = 2010;
trainName = 'train';
testName = 'val';

% Task
nnOpts.fastRcnn = false;
nnOpts.lossFnObjective = 'hinge';
nnOpts.testFn = @testClassification;
nnOpts.derOutputs = {'objective', 1};

% General
nnOpts.batchSize = 128;
nnOpts.numSubBatches = 1;
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = [repmat(1e-3, 12, 1); repmat(1e-4, 4, 1)];
nnOpts.misc.netPath = fullfile(MYDATADIR, 'MatconvnetModels', 'imagenet-vgg-verydeep-16.mat');
nnOpts.gpus = SelectIdleGpu();

% Setup data opts
setupDataOpts(vocYear, testName);
global DATAopts; % Database specific paths
nnOpts.expDir = [DATAopts.resultsPath, sprintf('FastRcnnMatconvnet/CalvinClassificationRun/')];

% Start from pretrained network
net = load(nnOpts.misc.netPath);

% Setup imdb
imdb = setupImdbClassification(trainName, testName, net);

% Create calvinNN CNN class. Note nnOpts above for classification
calvinn = CalvinNN(net, imdb, nnOpts);

% Create calvinNN CNN class. By default, network is transformed into fast-rcnn with bbox regression
calvinn = CalvinNN(net, imdb, nnOpts);

% Train
calvinn.train();

% Test
stats = calvinn.test();

% Eval
evalClassification();




if USEGPU
    exit
end
