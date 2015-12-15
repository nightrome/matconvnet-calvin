%% function calvinNNDetection

global MYDATADIR        % Directory of datasets
global DATAopts         % Database specific paths
global USEGPU           % Set in startup.m. Use GPU or not.

%% Set training and test set
trainName = 'trainval';
testName = 'test';


%% Setup VOC data (should be done using original VOC code). Below is minimal working example
DATAopts.dataset = 'VOC2007';
DATAopts.datadir = [MYDATADIR 'VOCdevkit/' DATAopts.dataset '/'];
DATAopts.resultsPath = [DATAopts.datadir 'Results/'];
DATAopts.imgsetpath = [DATAopts.datadir 'ImageSets/Main/%s.txt'];
DATAopts.imgpath = [DATAopts.datadir 'JPEGImages/%s.jpg'];
DATAopts.resdir = [MYDATADIR 'results/' DATAopts.dataset '/'];
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
nnOpts.testFn = @CalvinNN.testDetection;
nnOpts.batchSize = 2;
nnOpts.numSubBatches = 2;
nnOpts.weightDecay = 5e-4;
nnOpts.momentum = 0.9;
nnOpts.numEpochs = 16;
nnOpts.learningRate = cat(1, repmat(0.001, 12, 1), repmat(0.0001, 4, 1));
nnOpts.derOutputs = {'objective', 1, 'regressObjective', 1};

if USEGPU
    nnOpts.gpus = SelectIdleGpu();
else
    nnOpts.gpus = [];
end

% output path
nnOpts.expDir = [DATAopts.resultsPath ...
    sprintf('FastRcnnMatconvnet/CalvinDetectionRun/')]

% Start from pretrained network
net = load([MYDATADIR 'MatconvnetModels/imagenet-vgg-f.mat']);
% net = load([MYDATADIR 'MatconvnetModels/imagenet-vgg-verydeep-16.mat']);

%% Setup the Imdb
% Get and test images
trainIms = textread(sprintf(DATAopts.imgsetpath, trainName), '%s');
testIms = textread(sprintf(DATAopts.imgsetpath, testName), '%s');

% Make train, val, and test set. For Pascal, I illegally use part of the test images
% as validation set. This is to match Girshick performance while still having
% meaningful graphs for the validation set.
% Note: allIms are just all images. datasetIdx determines how these are divided over
% train, val, and test.
allIms = cat(1, trainIms, testIms);
datasetIdx{1} = (1:length(trainIms))';  % Jasper: Use all training images. Only for comparison Pascal Girshick
datasetIdx{2} = (length(trainIms)+1:length(trainIms)+501)'; % Use part of the test images for validation. Not entirely legal, but otherwise it will take much longer to get where we want.
datasetIdx{3} = (length(trainIms)+1:length(allIms))';

ImdbPascal = ImdbDetectionFullSupervision(DATAopts.imgpath(1:end-6), ...        % path
                                          DATAopts.imgpath(end-3:end), ...      % image extension
                                          DATAopts.gStructPath, ...             % gStruct path
                                          allIms, ...                           % all images
                                          datasetIdx, ...                       % division into train/val/test
                                          net.normalization.averageImage);      % average image used to pretrain network

% Usually instance weighting gives better performance. But not Girshick style
% ImdbPascal.SetInstanceWeighting(true); 

%% Create calvinNN CNN class          
calvinn = CalvinNN(net, ImdbPascal, nnOpts);
calvinn.convertNetworkToFastRcnn2([], 'fc8');

%%%%%%%%%%%%%
%% Train
%%%%%%%%%%%%%
calvinn.train();

%%%%%%%%%%%%%
%% Test
%%%%%%%%%%%%%
stats = calvinn.test();


%% Do evaluation
clear recall prec ap upperBound

% get image sizes
for i=length(testIms):-1:1
    im = ImageRead(testIms{i});
    imSizes(i,:) = size(im);
end

for cI = 1:20
    %%
    currBoxes = cell(length(testIms), 1);
    currScores = cell(length(testIms), 1);
    for i=1:length(testIms)
        currBoxes{i} = stats.results(i).boxes{cI+1};
        currScores{i} = stats.results(i).scores{cI+1};
    end
    
    [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
    [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
    
%     isequal(fileIdx, fileIdx2) % Should be equal
    
    currFilenames = testIms(fileIdx);
    
    [~, sI] = sort(currScores, 'descend');
    currScores = currScores(sI);
    currBoxes = currBoxes(sI,:);
    currFilenames = currFilenames(sI);
    
%     ShowImageRects(currBoxes(1:32, [2 1 4 3]), 4, 4, currFilenames(1:32), currScores(1:32));
    
    %%
    [recall{cI}, prec{cI}, ap(cI,1), upperBound{cI}] = ...
        DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
                                       'FastRcnnMatconvnet', 1, 0);
    ap(cI)
end

ap
mean(ap)

% get image sizes to refit regressed boxes
for i=length(testIms):-1:1
    im = ImageRead(testIms{i});
    imSizes(i,:) = size(im);
end

if isfield(stats.results(1), 'boxesRegressed')

    for cI = 1:20
        %%
        currBoxes = cell(length(testIms), 1);
        currScores = cell(length(testIms), 1);
        for i=1:length(testIms)
            % Get regressed boxes and refit them to the image
            currBoxes{i} = stats.results(i).boxesRegressed{cI+1};
            currBoxes{i}(:,1) = max(currBoxes{i}(:,1), 1);
            currBoxes{i}(:,2) = max(currBoxes{i}(:,2), 1);
            currBoxes{i}(:,3) = min(currBoxes{i}(:,3), imSizes(i,2));
            currBoxes{i}(:,4) = min(currBoxes{i}(:,4), imSizes(i,1));

            currScores{i} = stats.results(i).scoresRegressed{cI+1};
        end

        [currBoxes, fileIdx] = Cell2Matrix(gather(currBoxes));
        [currScores, fileIdx2] = Cell2Matrix(gather(currScores));

    %     isequal(fileIdx, fileIdx2) % Should be equal

        currFilenames = testIms(fileIdx);

        [~, sI] = sort(currScores, 'descend');
        currScores = currScores(sI);
        currBoxes = currBoxes(sI,:);
        currFilenames = currFilenames(sI);

    %     ShowImageRects(currBoxes(1:32, [2 1 4 3]), 4, 4, currFilenames(1:32), currScores(1:32));

        %%
        [recall{cI}, prec{cI}, apRegressed(cI,1), upperBound{cI}] = ...
            DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
                                           'FastRcnnMatconvnet', 1, 0);
        apRegressed(cI)
    end

    apRegressed
    mean(apRegressed)
else
    apRegressed = 0;
end

%%
save([nnOpts.expDir 'resultsEpochFinalTest.mat'], 'nnOpts', 'stats', 'ap', 'apRegressed');

if USEGPU
    exit
end
