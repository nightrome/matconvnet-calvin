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
initLinCombPath = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, 'fcn16s-notrain-ilsvrc-auto-lincomb-trn', 'linearCombination-trn.mat');
modelPathFunc = @(epoch) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', epoch));

% training options (SGD)
existingEpoch = CalvinNN.findLastCheckpoint(opts.expDir);
opts.train.batchSize = 20;
opts.train.numSubBatches = opts.train.batchSize;
opts.train.continue = existingEpoch > 0;
opts.train.gpus = gpus;
opts.train.prefetch = true;
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
    imdb.weaklySupervised = weaklySupervised;
    imdb.semiSupervised = semiSupervised;
    imdb.semiSupervisedRate = semiSupervisedRate;
    imdb.useInvFreqWeights = useInvFreqWeights;
    
    % Specify level of supervision for each train image
    if ~imdb.weaklySupervised
        % FS
        imdb.images.isFullySupervised = true(numel(imdb.images.name), 1);
    elseif ~imdb.semiSupervised
        % WS
        imdb.images.isFullySupervised = false(numel(imdb.images.name), 1);
    else
        % SS: Set x% of train and all val to true
        imdb.images.isFullySupervised = true(numel(imdb.images.name), 1);
        trainSel = find(imdb.images.set == 1);
        trainSel = trainSel(randperm(numel(trainSel)));
        trainSel = trainSel((trainSel / numel(trainSel)) >= semiSupervisedRate);
        imdb.images.isFullySupervised(trainSel) = false;
        
        if semiSupervisedOnlyFS
            % Keep x% of train and all val
            sel = imdb.images.isFullySupervised(:) | imdb.images.set(:) == 2;
            imdb.images.name = imdb.images.name(sel);
            imdb.images.set = imdb.images.set(sel);
            imdb.images.segmentation = imdb.images.segmentation(sel);
            imdb.images.isFullySupervised = imdb.images.isFullySupervised(sel);
            
            if strStartsWith(dataset.name, 'VOC')
                imdb.images.id = imdb.images.id(sel);
                imdb.images.classification = imdb.images.classification(sel);
                imdb.images.size = imdb.images.size(:, sel);
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
    
    if imdb.weaklySupervised
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
    layerSSInputs = [net.layers(objIdx).inputs, {'labelsImage', 'classWeights', 'isWeaklySupervised'}];
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
    if imdb.weaklySupervised,
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
fn = @(imdb,batch) getBatch(imdb, batch, opts, 'prefetch', nargout==0);

function y = getBatch(imdb, images, varargin)
% GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.imageSize = [512, 512] - 128;
opts.numAugments = 1;
opts.transformation = 'none';
opts.rgbMean = [];
opts.rgbVariance = zeros(0,3,'single');
opts.labelStride = 1;
opts.labelOffset = 1;
opts.classWeights = [];
opts.interpolation = 'bilinear';
opts.prefetch = false;
opts.useGpu = false;
opts.imageNameToLabelMap = @(imageName, imdb) imread(sprintf(imdb.paths.classSegmentation, imageName));
opts.translateLabels = true;
opts = vl_argparse(opts, varargin);

if opts.prefetch
    % to be implemented
    y = {};
    return;
end

imageCount = numel(images);
augmentCount = imageCount * opts.numAugments;
assert(imageCount == 1);

if ~isempty(opts.rgbVariance) && isempty(opts.rgbMean)
    opts.rgbMean = single([128;128;128]);
end
if ~isempty(opts.rgbMean)
    opts.rgbMean = reshape(opts.rgbMean, [1 1 3]);
end

% space for images
ims = zeros(opts.imageSize(1), opts.imageSize(2), 3, augmentCount, 'single');

% space for labels
lx = opts.labelOffset : opts.labelStride : opts.imageSize(2);
ly = opts.labelOffset : opts.labelStride : opts.imageSize(1);
labels = zeros(numel(ly), numel(lx), 1, augmentCount, 'double'); % must be double for to avoid numerical precision errors in vl_nnloss, when using many classes
if imdb.weaklySupervised,
    labelsImageCell = cell(augmentCount, 1);
end;

si = 1;

for i = 1 : imageCount
    % acquire image
    imageName = imdb.images.name{images(i)};
    rgb = double(imdb.dataset.getImage(imageName)) * 255;
    if size(rgb,3) == 1
        rgb = cat(3, rgb, rgb, rgb);
    end
    
    % acquire pixel-level GT
    if imdb.dataset.annotation.hasPixelLabels,
        anno = uint16(opts.imageNameToLabelMap(imageName, imdb));
        
        % Translate labels s.t. 255 is mapped to 0
        if opts.translateLabels,
            % Before: 255 = ignore, 0 = bkg, 1:labelCount = classes
            % After : 0 = ignore, 1 = bkg, 2:labelCount+1 = classes
            anno = mod(anno + 1, 256);
        end;
        % 0 = ignore, 1:labelCount = classes
    else
        anno = [];
    end;
    
    % crop & flip
    h = size(rgb,1);
    w = size(rgb,2);
    for ai = 1:opts.numAugments
        sz = opts.imageSize(1:2);
        scale = max(h/sz(1), w/sz(2));
        scale = scale .* (1 + (rand(1)-.5)/5);
        
        sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2);
        sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2);
        if rand > 0.5, sx = fliplr(sx); end
        
        okx = find(1 <= sx & sx <= w);
        oky = find(1 <= sy & sy <= h);
        if ~isempty(opts.rgbMean)
            ims(oky, okx, :, si) = bsxfun(@minus, rgb(sy(oky), sx(okx), :), opts.rgbMean);
        else
            ims(oky, okx, :, si) = rgb(sy(oky), sx(okx),:);
        end
        
        % Fully supervised: Get pixel level labels
        if ~isempty(anno)
            tlabels = zeros(sz(1), sz(2), 'double');
            tlabels(oky,okx) = anno(sy(oky), sx(okx));
            tlabels = single(tlabels(ly,lx));
            labels(:, :, 1, si) = tlabels; % 0: ignore
        end;
        
        % Weakly supervised: extract image-level labels
        if imdb.weaklySupervised,
            if ~isempty(anno),
                % Get image labels from pixel labels
                % These are already translated (if necessary)
                curLabelsImage = unique(anno);
            else
                curLabelsImage = imdb.dataset.getImLabelInds(imageName);
                
                % Translate labels s.t. 255 is mapped to 0
                if opts.translateLabels
                    curLabelsImage = mod(curLabelsImage + 1, 256);
                end
                
                if imdb.dataset.annotation.labelOneIsBg
                    % Add background label
                    curLabelsImage = unique([0; curLabelsImage(:)]);
                end
            end;
            
            % Remove invalid pixels
            curLabelsImage(curLabelsImage == 0) = [];
            
            % Store image-level labels
            labelsImageCell{si} = single(curLabelsImage(:));
        end;
        
        si = si + 1;
    end
end

% Move image to GPU
if opts.useGpu
    ims = gpuArray(ims);
end

%%% Create outputs
y = {'input', ims};
if imdb.dataset.annotation.hasPixelLabels
    y = [y, {'label', labels}];
end
if imdb.weaklySupervised
    assert(~any(cellfun(@(x) isempty(x), labelsImageCell)));
    y = [y, {'labelsImage', labelsImageCell}];
end

% Instance/pixel weights, can be left empty
y = [y, {'classWeights', opts.classWeights}];

% Decide which level of supervision to pick
if imdb.semiSupervised
    % SS
    isWeaklySupervised = ~imdb.images.isFullySupervised(images);
else
    % FS or WS
    isWeaklySupervised = imdb.weaklySupervised;
end
if ~isWeaklySupervised
    assert(imdb.dataset.annotation.hasPixelLabels);
end
y = [y, {'isWeaklySupervised', isWeaklySupervised}];