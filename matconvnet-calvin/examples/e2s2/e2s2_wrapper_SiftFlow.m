 function e2s2_wrapper_SiftFlow()
% e2s2_wrapper_SiftFlow()
%
% A wrapper for Fast-RCNN with Matconvnet that allows to train and test a network.
%
% Copyright by Holger Caesar, 2015

% Settings
global glFeaturesFolder;
projectName = 'WeaklySupervisedLearning';
run = 28;
exp = 0;
netName = 'VGG16';
dataset = SiftFlowDataset();
gpus = [];
roiPool.use = true;
roiPool.freeform.use = true;
roiPool.freeform.combineFgBox = true;
roiPool.freeform.shareWeights = false;
regionToPixel.use = true;
regionToPixel.minPixFreq = [];
regionToPixel.inverseLabelFreqs = true;
regionToPixel.replicateUnpureSPs = true;
regionToPixel.normalizeImageMass = true;
trainValRatio = 0.9;
randSeed = 280;
logFile = 'log.txt';
batchSize = 10;
numEpochs = 30; % makes sure we always do the same number of gradient steps
highLRNumEpochs = 20;
lowLRNumEpochs  = numEpochs - highLRNumEpochs;
highLR = repmat(1e-3, [1, highLRNumEpochs]);
lowLR  = repmat(1e-4, [1,  lowLRNumEpochs]);
learningRate = [highLR, lowLR];
segments.minSize = 100;
segments.switchColorTypesEpoch = true;
segments.switchColorTypesBatch = true;
segments.colorTypes = {'Rgb', 'Hsv', 'Lab'};
segments.colorTypeIdx = 1;
fastRcnnParams = false;

% Initialize random number generator seed
if ~isempty(randSeed);
    rng(randSeed);
    if ~isempty(gpus),
        randStream = parallel.gpu.RandStream('CombRecursive', 'Seed', randSeed);
        parallel.gpu.RandStream.setGlobalStream(randStream);
    end;
end;

% Create paths
if strcmp(netName, 'AlexNet'),
    netFileName = 'imagenet-caffe-alex';
elseif strcmp(netName, 'VGG16'),
    netFileName = 'imagenet-vgg-verydeep-16';
elseif strcmp(netName, '18beta-AlexNet'),
    netFileName = '18beta/imagenet-matconvnet-alex';
elseif strcmp(netName, '18beta-VGG16'),
    netFileName = '18beta/imagenet-matconvnet-vgg-verydeep-16';
else
    error('Error: Unknown netName!');
end;
outputFolderName = sprintf('%s_finetune_e2s2_%s_run%d_exp%d', dataset.name, netName, run, exp);
netPath = fullfile(glFeaturesFolder, 'CNN-Models', 'matconvnet', [netFileName, '.mat']);
segmentFolder = fullfile(glFeaturesFolder, projectName, dataset.name, 'segmentations');
outputFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'E2S2', dataset.name, sprintf('Run%d', run), outputFolderName);

% Create outputFolder
if ~exist(outputFolder, 'dir'),
    mkdir(outputFolder);
end;

% Start logging
if ~isempty(logFile),
    diary(fullfile(outputFolder, logFile));
end;

% Get images
imageListTrn = dataset.getImageListTrn(true);
imageListTst = dataset.getImageListTst(true);

% Store in imdb
imdb = ImdbE2S2(dataset, segmentFolder);
imdb.dataset = dataset;
trainValList = rand(numel(imageListTrn), 1) <= trainValRatio;
imdb.data.train = imageListTrn( trainValList);
imdb.data.val   = imageListTrn(~trainValList);
imdb.data.test  = imageListTst;
imdb.numClasses = dataset.labelCount;
imdb.batchOpts.segments = structOverwriteFields(imdb.batchOpts.segments, segments);
imdb.updateSegmentNames();

% Create nnOpts
nnOpts = struct();
nnOpts.expDir = outputFolder;
nnOpts.numEpochs = numEpochs;
nnOpts.batchSize = batchSize;
nnOpts.numSubBatches = batchSize; % Always the same as batchSize!
nnOpts.gpus = gpus;
nnOpts.continue = CalvinNN.findLastCheckpoint(outputFolder) > 0;
nnOpts.learningRate = learningRate;
nnOpts.extractStatsFn = @E2S2NN.extractStats;
nnOpts.misc.roiPool = roiPool;
nnOpts.misc.regionToPixel = regionToPixel;
nnOpts.bboxRegress = false;
nnOpts.fastRcnnParams = fastRcnnParams;

% Save the current options
netOptsPath = fullfile(outputFolder, 'net-opts.mat');
if exist(netOptsPath, 'file'),
    % Make sure the store options correspond to the current ones
    netOptsOld = load(netOptsPath, 'nnOpts', 'imdb');
    assert(isequal(netOptsOld.nnOpts.learningRate, nnOpts.learningRate(1:netOptsOld.nnOpts.numEpochs)));
    assert(isEqualFuncs(netOptsOld.nnOpts, nnOpts, {'continue', 'gpus', 'learningRate', 'numEpochs', 'expDir'}));
    assert(isEqualFuncs(netOptsOld.imdb, imdb, {}));
else
    save(netOptsPath, 'nnOpts', 'imdb', '-v6');
end;

% Create network
nnClass = E2S2NN(netPath, imdb, nnOpts);

% Train the network
nnClass.train();

% Test the network
stats = nnClass.test();
disp(stats);

% Save the results
statsPath = fullfile(outputFolder, sprintf('stats-epoch-%d.mat', nnClass.nnOpts.numEpochs));
if exist(statsPath, 'file'),
    error('Error: statsPath already exists: %s', statsPath);
else
    save(statsPath, 'stats', '-v6');
end;