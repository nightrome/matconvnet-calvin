function fcnTrainGeneric(varargin)
% fcnTrainGeneric(varargin)
%
% Train FCN model using MatConvNet.
%
% Copyright by Holger Caesar, 2016

% Initial settings
p = inputParser;
addParameter(p, 'dataset', SiftFlowDataset());
addParameter(p, 'modelType', 'fcn16s');
addParameter(p, 'modelFile', '19beta/imagenet-vgg-verydeep-16.mat');
addParameter(p, 'gpus', 2);
addParameter(p, 'randSeed', 42);
addParameter(p, 'expNameAppend', 'test');
addParameter(p, 'weaklySupervised', false);
addParameter(p, 'numEpochs', 50);
addParameter(p, 'useInvFreqWeights', false);
addParameter(p, 'wsUseAbsent', false);      % helpful
addParameter(p, 'wsUseScoreDiffs', false);  % not helpful
addParameter(p, 'wsEqualWeight', false);    % not helpful
addParameter(p, 'semiSupervised', false);
addParameter(p, 'semiSupervisedRate', 0.1);     % ratio of images with full supervision
addParameter(p, 'semiSupervisedOnlyFS', false); % use only the x% fully supervised images
addParameter(p, 'init', 'zeros'); % zeros, best-auto, best-manual, lincomb (all +-autobias)
addParameter(p, 'enableCudnn', false);
addParameter(p, 'maskThings', false);
parse(p, varargin{:});

dataset = p.Results.dataset;
modelType = p.Results.modelType;
modelFile = p.Results.modelFile;
gpus = p.Results.gpus;
randSeed = p.Results.randSeed;
expNameAppend = p.Results.expNameAppend;
weaklySupervised = p.Results.weaklySupervised;
numEpochs = p.Results.numEpochs;
useInvFreqWeights = p.Results.useInvFreqWeights;
wsUseAbsent = p.Results.wsUseAbsent;
wsUseScoreDiffs = p.Results.wsUseScoreDiffs;
wsEqualWeight = p.Results.wsEqualWeight;
semiSupervised = p.Results.semiSupervised;
semiSupervisedRate = p.Results.semiSupervisedRate;
semiSupervisedOnlyFS = p.Results.semiSupervisedOnlyFS;
init = p.Results.init;
enableCudnn = p.Results.enableCudnn;
maskThings = p.Results.maskThings;
callArgs = p.Results; %#ok<NASGU>

% Check settings for consistency
if semiSupervised
    assert(weaklySupervised);
end
if isa(dataset, 'VOC2011Dataset')
    assert(~useInvFreqWeights);
end;

% experiment and data paths
global glBaseFolder glFeaturesFolder;
dataRootDir = fullfile(glBaseFolder, 'CodeForeign', 'CNN', 'matconvnet-fcn', 'data');
expName = [modelType, prependNotEmpty(expNameAppend, '-')];
opts.expDir = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, expName);
opts.sourceModelPath = fullfile(glFeaturesFolder, 'CNN-Models', 'matconvnet', modelFile);
logFilePath = fullfile(opts.expDir, 'log.txt');
initLinCombPath = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, 'notrain', 'fcn16s-notrain-ilsvrc-lincomb-trn', 'linearCombination-trn.mat');
modelPathFunc = @(epoch) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', epoch));

% training options (SGD)
existingEpoch = CalvinNN.findLastCheckpoint(opts.expDir);
opts.train.batchSize = 20;
opts.train.numSubBatches = opts.train.batchSize;
opts.train.continue = existingEpoch > 0;
opts.train.gpus = gpus;
opts.train.prefetch = false;
opts.train.expDir = opts.expDir;
opts.train.numEpochs = numEpochs;
opts.train.learningRate = 1e-4;
opts.modelType = modelType;

% Fix randomness
rng(randSeed);

% Create folders
if ~exist(opts.expDir, 'dir'),
    mkdir(opts.expDir);
end;

% Setup logfile
diary(logFilePath);

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
else
    %%% VOC specific
    if strStartsWith(dataset.name, 'VOC'),
        % Get PASCAL VOC segmentation dataset plus Berkeley's additional segmentations
        opts.vocAdditionalSegmentations = true;
        opts.vocEdition = '11';
        opts.dataDir = fullfile(dataRootDir, dataset.name);
        imdb = vocSetup('dataDir', opts.dataDir, ...
            'edition', opts.vocEdition, ...
            'includeTest', false, ...
            'includeSegmentation', true, ...
            'includeDetection', false);
        if opts.vocAdditionalSegmentations
            imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir);
        end
        
        stats = getDatasetStatistics(imdb);
        imdb.rgbMean = stats.rgbMean;
        imdb.translateLabels = true;
        imdb.imageNameToLabelMap = @(imageName, imdb) imread(sprintf(imdb.paths.classSegmentation, imageName));
    else
        %%% Other datasets
        % Get labels and image path
        imdb.classes.name = dataset.getLabelNames();
        imdb.paths.image = fullfile(dataset.getImagePath(), sprintf('%%s%s', dataset.imageExt));
        
        % Get trn + tst/val images
        imageListTrn = dataset.getImageListTrn();
        imageListTst = dataset.getImageListTst();
        
        % Remove images without labels
        missingImageIndicesTrn = dataset.getMissingImageIndices('train');
        imageListTrn(missingImageIndicesTrn) = [];
        % TODO: is it a good idea to remove test images?
        % (only doing it on non-competitive EdiStuff
        if isa(dataset, 'EdiStuffDataset') || isa(dataset, 'EdiStuffSubsetDataset')
            missingImageIndicesTst = dataset.getMissingImageIndices('test');
            imageListTst(missingImageIndicesTst) = [];
        end
        imageCountTrn = numel(imageListTrn);
        imageCountTst = numel(imageListTst);
        
        imdb.images.name = [imageListTrn; imageListTst];
        imdb.images.segmentation = true(imageCountTrn+imageCountTst, 1);
        imdb.images.set = nan(imageCountTrn+imageCountTst, 1);
        imdb.images.set(1:imageCountTrn) = 1;
        imdb.images.set(imageCountTrn+1:end) = 2;
        
        imdb.rgbMean = dataset.getMeanColor();
        imdb.translateLabels = false;
        imdb.imageNameToLabelMap = @(imageName, imdb) imdb.dataset.getImLabelMap(imageName);
    end;
    
    % Dataset-independent imdb fields
    imdb.dataset = dataset;
    imdb.labelCount = dataset.labelCount;
    
    % Specify level of supervision for each train image
    if ~weaklySupervised
        % FS
        imdb.images.isFullySupervised = true(numel(imdb.images.name), 1);
    elseif ~semiSupervised
        % WS
        imdb.images.isFullySupervised = false(numel(imdb.images.name), 1);
    else
        % SS: Set x% of train and all val to true
        imdb.images.isFullySupervised = true(numel(imdb.images.name), 1);
        if isa(imdb.dataset, 'EdiStuffDataset')
            selWS = find(~ismember(imdb.images.name, imdb.dataset.datasetFS.getImageListTrn()));
            assert(numel(selWS) == 18431);
        else
            selTrain = find(imdb.images.set == 1);
            selTrain = selTrain(randperm(numel(selTrain)));
            selWS = selTrain((selTrain / numel(selTrain)) >= semiSupervisedRate);
        end
        imdb.images.isFullySupervised(selWS) = false;
        
        if semiSupervisedOnlyFS
            % Keep x% of train and all val
            selFS = imdb.images.isFullySupervised(:) | imdb.images.set(:) == 2;
            imdb.images.name = imdb.images.name(selFS);
            imdb.images.set = imdb.images.set(selFS);
            imdb.images.segmentation = imdb.images.segmentation(selFS);
            imdb.images.isFullySupervised = imdb.images.isFullySupervised(selFS);
            
            if strStartsWith(dataset.name, 'VOC')
                imdb.images.id = imdb.images.id(selFS);
                imdb.images.classification = imdb.images.classification(selFS);
                imdb.images.size = imdb.images.size(:, selFS);
            end
        end
    end
    
    % Make sure val images are always fully supervised
    imdb.images.isFullySupervised(imdb.images.set == 2) = true;
    
    % Print overview of the fully and weakly supervised number of training
    % images
    fsCount = sum( imdb.images.isFullySupervised(:) & imdb.images.set(:) == 1);
    wsCount = sum(~imdb.images.isFullySupervised(:) & imdb.images.set(:) == 1);
    fsRatio = fsCount / (fsCount+wsCount);
    wsRatio = 1 - fsRatio;
    fprintf('Images in train: %d FS (%.1f%%), %d WS (%.1f%%)...\n', fsCount, fsRatio * 100, wsCount, wsRatio * 100);
    
    % Save imdb
    save(opts.imdbPath, '-struct', 'imdb');
end;

% Get training and test/validation subsets
% We always validate and test on val
train = find(imdb.images.set == 1 & imdb.images.segmentation);
val   = find(imdb.images.set == 2 & imdb.images.segmentation);

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

if existingEpoch == 0
    % Load an existing model
    netStruct = load(modelPathFunc(existingEpoch), 'net');
    net = dagnn.DagNN.loadobj(netStruct.net);
    clearvars netStruct;
elseif existingEpoch > 0
    net = {};
elseif isnan(existingEpoch)
    % Get initial model from VGG-VD-16
    net = fcnInitializeModelGeneric(imdb, 'sourceModelPath', opts.sourceModelPath, 'init', init, 'initLinCombPath', initLinCombPath, 'enableCudnn', enableCudnn);
    if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
        % upgrade model to FCN16s
        net = fcnInitializeModel16sGeneric(imdb.labelCount, net);
    end
    if strcmp(opts.modelType, 'fcn8s')
        % upgrade model fto FCN8s
        net = fcnInitializeModel8sGeneric(imdb.labelCount, net);
    end
    net.meta.normalization.rgbMean = imdb.rgbMean;
    net.meta.classes = imdb.classes.name;
    
    if weaklySupervised
        wsPresentWeight = 1 / (1 + wsUseAbsent);
        
        if wsEqualWeight
            wsAbsentWeight = imdb.labelCount * wsUseAbsent; % TODO: try -log(2/21) / -log(1-2/21)
        else
            wsAbsentWeight = 1 - wsPresentWeight;
        end
    else
        wsPresentWeight = [];
        wsAbsentWeight = [];
    end
    
    % Replace unweighted loss layer
    layerFS = dagnn.SegmentationLossPixel();
    layerWS = dagnn.SegmentationLossImage('useAbsent', wsUseAbsent, 'useScoreDiffs', wsUseScoreDiffs, 'presentWeight', wsPresentWeight, 'absentWeight', wsAbsentWeight);
    objIdx = net.getLayerIndex('objective');
    assert(strcmp(net.layers(objIdx).block.loss, 'softmaxlog'));
    
    % Add a layer that automatically decides whether to use FS or WS
    layerSS = dagnn.SegmentationLossSemiSupervised('layerFS', layerFS, 'layerWS', layerWS);
    layerSSInputs = [net.layers(objIdx).inputs, {'labelsImage', 'classWeights', 'isWeaklySupervised', 'masksThingsCell'}];
    layerSSOutputs = net.layers(objIdx).outputs;
    net.removeLayer('objective');
    net.addLayer('objective', layerSS, layerSSInputs, layerSSOutputs, {});
    
    % Accuracy layer
    if imdb.dataset.annotation.hasPixelLabels
        % Replace accuracy layer with 21 classes by flexible accuracy layer
        accIdx = net.getLayerIndex('accuracy');
        accLayer = net.layers(accIdx);
        accInputs = accLayer.inputs;
        accOutputs = accLayer.outputs;
        accBlock = dagnn.SegmentationAccuracyFlexible('labelCount', imdb.labelCount);
        net.removeLayer('accuracy');
        net.addLayer('accuracy', accBlock, accInputs, accOutputs, {});
    else
        % Remove accuracy layer if no pixel-level labels exist
        net.removeLayer('accuracy');
    end
end

% Extract inverse class frequencies from dataset
if useInvFreqWeights,
    if weaklySupervised,
        classWeights = imdb.dataset.getLabelImFreqs('train');
    else
        classWeights = imdb.dataset.getLabelPixelFreqs('train');
    end;
    
    % Inv freq and normalize classWeights
    classWeights = classWeights ./ sum(classWeights);
    nonEmpty = classWeights ~= 0;
    classWeights(nonEmpty) = 1 ./ classWeights(nonEmpty);
    classWeights = classWeights ./ sum(classWeights);
    assert(~any(isnan(classWeights)));
else
    classWeights = [];
end;

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.classWeights = classWeights;
bopts.rgbMean = imdb.rgbMean;
bopts.useGpu = numel(opts.train.gpus) > 0;
bopts.imageNameToLabelMap = imdb.imageNameToLabelMap;
bopts.translateLabels = imdb.translateLabels;
bopts.maskThings = maskThings;
bopts.weaklySupervised = weaklySupervised;
bopts.semiSupervised = semiSupervised;
bopts.useInvFreqWeights = useInvFreqWeights;

% Save important settings
settingsPath = fullfile(opts.expDir, 'settings.mat');
save(settingsPath, 'callArgs', 'opts', 'bopts');

% Save net before training
if isnan(existingEpoch)
    saveStruct.net = net.saveobj();
    saveStruct.stats = []; %#ok<STRNU>
    modelPath = modelPathFunc(0);
    assert(~exist(modelPath, 'file'));
    save(modelPath, '-struct', 'saveStruct');
    clearvars saveStruct;
end

% Launch SGD
[~, stats] = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', train, ...
    'val', val); %#ok<ASGLU>

% Output stats
statsPath = fullfile(opts.expDir, 'stats.mat');
save(statsPath, 'stats');

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb, batch) getBatch(imdb, batch, opts);

function y = getBatch(imdb, imageInds, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation
%
% Note: Currently train and val batches are treated the same.

bopts.imageSize = [512, 512] - 128;
bopts.rgbMean = single([128; 128; 128]);
bopts.labelStride = 1;
bopts.classWeights = [];
bopts.useGpu = false;
bopts.imageNameToLabelMap = @(imageName, imdb) imread(sprintf(imdb.paths.classSegmentation, imageName));
bopts.translateLabels = true;
bopts.maskThings = false;
bopts.weaklySupervised = false;
bopts.semiSupervised = false;
bopts.useInvFreqWeights = false;
bopts = vl_argparse(bopts, varargin);

% Check settings
assert(~isempty(bopts.rgbMean));
bopts.rgbMean = reshape(bopts.rgbMean, [1 1 3]);

% Make sure that the subbatch size is one image
imageCount = numel(imageInds);
if imageCount == 0
    % Empty batch
    y = {};
    return;
elseif imageCount == 1
    % Default
else
    error('Error: GetBatch cannot process more than 1 image at a time!');
end
imageIdx = imageInds;

% Initializations
ims = zeros(bopts.imageSize(1), bopts.imageSize(2), 3, imageCount, 'single');
lx = 1 : bopts.labelStride : bopts.imageSize(2);
ly = 1 : bopts.labelStride : bopts.imageSize(1);
labels = zeros(numel(ly), numel(lx), 1, imageCount, 'double'); % must be double for to avoid numerical precision errors in vl_nnloss, when using many classes
if bopts.weaklySupervised
    labelsImageCell = cell(imageCount, 1);
end
if bopts.maskThings
    assert(isa(imdb.dataset, 'EdiStuffDataset'));
    datasetIN = ImageNetDataset();
end
masksThingsCell = cell(imageCount, 1); % by default this is empty

if true
    % Get image
    imageName = imdb.images.name{imageIdx};
    rgb = double(imdb.dataset.getImage(imageName)) * 255;
    if size(rgb,3) == 1
        rgb = cat(3, rgb, rgb, rgb);
    end
    
    % Get pixel-level GT
    if imdb.dataset.annotation.hasPixelLabels || imdb.images.isFullySupervised(imageIdx)
        anno = uint16(bopts.imageNameToLabelMap(imageName, imdb));
        
        % Translate labels s.t. 255 is mapped to 0
        if bopts.translateLabels,
            % Before: 255 = ignore, 0 = bkg, 1:labelCount = classes
            % After : 0 = ignore, 1 = bkg, 2:labelCount+1 = classes
            anno = mod(anno + 1, 256);
        end;
        % 0 = ignore, 1:labelCount = classes
    else
        anno = [];
    end;
    
    % Crop and rescale image
    h = size(rgb, 1);
    w = size(rgb, 2);
    sz = bopts.imageSize(1 : 2);
    scale = max(h / sz(1), w / sz(2));
    scale = scale .* (1 + (rand(1) - .5) / 5);
    sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2);
    sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2);
    
    % Flip image
    if rand > 0.5
        sx = fliplr(sx);
    end
    
    % Get image indices in valid area
    okx = find(1 <= sx & sx <= w);
    oky = find(1 <= sy & sy <= h);
    
    % Subtract mean image
    ims(oky, okx, :, 1) = bsxfun(@minus, rgb(sy(oky), sx(okx), :), bopts.rgbMean);
    
    % Fully supervised: Get pixel level labels
    if ~isempty(anno)
        tlabels = zeros(sz(1), sz(2), 'double');
        tlabels(oky,okx) = anno(sy(oky), sx(okx));
        tlabels = single(tlabels(ly,lx));
        labels(:, :, 1, 1) = tlabels; % 0: ignore
    end;
    
    % Weakly supervised: extract image-level labels
    if bopts.weaklySupervised
        if ~isempty(anno) && ~all(anno(:) == 0)
            % Get image labels from pixel labels
            % These are already translated (if necessary)
            curLabelsImage = unique(anno);
        else
            curLabelsImage = imdb.dataset.getImLabelInds(imageName);
            
            % Translate labels s.t. 255 is mapped to 0
            if bopts.translateLabels
                curLabelsImage = mod(curLabelsImage + 1, 256);
            end
            
            if imdb.dataset.annotation.labelOneIsBg
                % Add background label
                curLabelsImage = unique([0; curLabelsImage(:)]);
            end
        end
        
        % Remove invalid pixels
        curLabelsImage(curLabelsImage == 0) = [];
        
        % Store image-level labels
        labelsImageCell{1} = single(curLabelsImage(:));
    end
    
    % Optional: Mask out thing pixels
    if bopts.maskThings
        % Get mask
        longName = datasetIN.shortNameToLongName(imageName);
        mask = datasetIN.getImLabelBoxesMask(longName);
        
        % Resize it if necessary
        if size(mask, 1) ~= size(ims, 1) || ...
                size(mask, 2) ~= size(ims, 2)
            mask = imresize(mask, [size(ims, 1), size(ims, 2)]);
        end
        masksThingsCell{1} = mask;
    end
end

% Move image to GPU
if bopts.useGpu
    ims = gpuArray(ims);
end

%%% Create outputs
y = {'input', ims};
if imdb.dataset.annotation.hasPixelLabels || imdb.images.isFullySupervised(imageIdx)
    y = [y, {'label', labels}];
end
if bopts.weaklySupervised
    assert(~any(cellfun(@(x) isempty(x), labelsImageCell)));
    y = [y, {'labelsImage', labelsImageCell}];
end

% Instance/pixel weights, can be left empty
y = [y, {'classWeights', bopts.classWeights}];

% Decide which level of supervision to pick
if bopts.semiSupervised
    % SS
    isWeaklySupervised = ~imdb.images.isFullySupervised(imageInds);
else
    % FS or WS
    isWeaklySupervised = bopts.weaklySupervised;
end
y = [y, {'isWeaklySupervised', isWeaklySupervised}];
y = [y, {'masksThingsCell', masksThingsCell}];