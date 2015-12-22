%% function calvinNNDetection

% Example for classification on Pascal.

global MYDATADIR        % Directory of datasets
global DATAopts         % Database specific paths
global USEGPU           % Set in startup.m. Use GPU or not.

%% Set training and test set
trainName = 'trainval';
testName = 'test';


%% Setup VOC data (should be done using original VOC code). Below is minimal working example
DATAopts.year = 2007;
DATAopts.dataset = 'VOC2007';
DATAopts.datadir = [MYDATADIR 'VOCdevkit/' DATAopts.dataset '/'];
DATAopts.resultsPath = [DATAopts.datadir 'Results/'];
DATAopts.imgsetpath = [DATAopts.datadir 'ImageSets/Main/%s.txt'];
DATAopts.clsimgsetpath=[DATAopts.datadir '/ImageSets/Main/%s_%s.txt'];
DATAopts.imgpath = [DATAopts.datadir 'JPEGImages/%s.jpg'];
DATAopts.resdir = [MYDATADIR 'VOCdevkit/results/' DATAopts.dataset '/'];
DATAopts.annopath = [DATAopts.datadir 'VOCdevkit/Annotations/%s.xml'];
DATAopts.annocachepath = [MYDATADIR 'VOCdevkit/local/VOC2007/%s_anno.mat'];
DATAopts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
DATAopts.nclasses = length(DATAopts.classes);
DATAopts.testset = testName;
DATAopts.minoverlap=0.5;

DATAopts.gStructPath = [DATAopts.resultsPath 'GStructs/'];


%% Options for CNN training
nnOpts.fastRcnn = false;
nnOpts.lossFnObjective = 'hinge';
nnOpts.testFn = @CalvinNN.testClassification;
nnOpts.batchSize = 128;
nnOpts.numSubBatches = 1;
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = cat(1, repmat(0.001, 12, 1), repmat(0.0001, 4, 1));

nnOpts.derOutputs = {'objective', 1};

if USEGPU
    nnOpts.gpus = SelectIdleGpu();
else
    nnOpts.gpus = [];
end

% output path
nnOpts.expDir = [DATAopts.resultsPath ...
    sprintf('FastRcnnMatconvnet/CalvinClassificationRun/')]

% Start from pretrained network
net = load([MYDATADIR 'MatconvnetModels/imagenet-vgg-f.mat']);
% net = load([MYDATADIR 'MatconvnetModels/imagenet-vgg-verydeep-16.mat']);

%% Setup the Imdb
% Get images and labels
set = trainName;
[trainIms, ~] = textread(sprintf(DATAopts.imgsetpath,set),'%s %d');
trainLabs = zeros(length(trainIms), DATAopts.nclasses);

% Get all testlabels
if (~strcmp(set,'test') || DATAopts.year == 2007)
    for classIdx = 1:DATAopts.nclasses
        class = DATAopts.classes{classIdx};
        [~, trainLabs(:,classIdx)] = textread(sprintf(DATAopts.clsimgsetpath,class,set),'%s %d');
    end

    trainLabs(trainLabs == 0) = -1;
end

set = testName;
[testIms, ~] = textread(sprintf(DATAopts.imgsetpath,set),'%s %d');
testLabs = zeros(length(testIms), DATAopts.nclasses);

% Get all testlabels
if (~strcmp(set,'test') || DATAopts.year == 2007)
    for classIdx = 1:DATAopts.nclasses
        class = DATAopts.classes{classIdx};
        [~, testLabs(:,classIdx)] = textread(sprintf(DATAopts.clsimgsetpath,class,set),'%s %d');
    end

    testLabs(testLabs == -1) = 0;
end

% Make train, val, and test set.
numValIms = 500;
allIms = cat(1, trainIms, testIms);
allLabs = cat(1, trainLabs, testLabs);
datasetIdx{1} = (1:length(trainIms)-numValIms)';  % Last numValIms are val set
datasetIdx{2} = (length(trainIms)-numValIms+1:length(trainIms))';
datasetIdx{3} = (length(trainIms)+1:length(allIms))';

% Setup the classification imdb
ImdbPascal = ImdbClassification(DATAopts.imgpath(1:end-6), ...        % path
                                DATAopts.imgpath(end-3:end), ...      % image extension
                                allIms, ...                           % all images
                                allLabs, ...                          % all labels
                                datasetIdx, ...                       % division into train/val/test
                                net.normalization.averageImage, ...   % average image used to pretrain network
                                20);                                  % num classes

%% Create calvinNN CNN class. Note nnOpts above for classification
calvinn = CalvinNN(net, ImdbPascal, nnOpts);

%%%%%%%%%%%%%
%% Train
%%%%%%%%%%%%%
calvinn.train();

%%%%%%%%%%%%%
%% Test
%%%%%%%%%%%%%
stats = calvinn.test();


%% Do evaluation
scores = zeros(size(testLabs));
for i=1:length(stats.results)
    scores(i,:) = stats.results(i).scores;
end
[map, ap, conf] = MeanAveragePrecision(scores, testLabs);

ap
map

%%
save([nnOpts.expDir 'resultsEpochFinalTest.mat'], 'nnOpts', 'stats', 'ap', 'conf');

if USEGPU
    exit
end
