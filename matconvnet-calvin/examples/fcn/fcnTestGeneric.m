function[info] = fcnTestGeneric(varargin)
% [info] = fcnTestGeneric(varargin)
%
% Copyright by Holger Caesar, 2016

% Initial settings
p = inputParser;
addParameter(p, 'dataset', VOC2011Dataset());
addParameter(p, 'modelType', 'fcn16s');
addParameter(p, 'gpus', 1);
addParameter(p, 'randSeed', 42);
addParameter(p, 'expNameAppend', 'test');
addParameter(p, 'epoch', 50);
addParameter(p, 'modelFamily', 'matconvnet');
addParameter(p, 'plotFreq', 15);
addParameter(p, 'printFreq', 30);
addParameter(p, 'showPlot', false);
addParameter(p, 'maxImSize', 700);
addParameter(p, 'doOutputMaps', false); % Optional: output predicted labels to file
addParameter(p, 'doCache', true);
addParameter(p, 'subset', 'tst'); % trn / tst
addParameter(p, 'expSubFolder', '');
addParameter(p, 'findMapping', false); % Find the best possible label mapping from ILSVRC to target dataset
parse(p, varargin{:});

dataset = p.Results.dataset;
modelType = p.Results.modelType;
gpus = p.Results.gpus;
randSeed = p.Results.randSeed;
expNameAppend = p.Results.expNameAppend;
epoch = p.Results.epoch;
modelFamily = p.Results.modelFamily;
plotFreq = p.Results.plotFreq;
printFreq = p.Results.printFreq;
maxImSize = p.Results.maxImSize;
showPlot = p.Results.showPlot;
doOutputMaps = p.Results.doOutputMaps;
doCache = p.Results.doCache;
subset = p.Results.subset;
expSubFolder = p.Results.expSubFolder;
findMapping = p.Results.findMapping;
callArgs = p.Results; %#ok<NASGU>

% experiment and data paths
global glFeaturesFolder;
expName = [modelType, prependNotEmpty(expNameAppend, '-')];
opts.expDir = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, expSubFolder, expName);
opts.modelPath = fullfile(opts.expDir, sprintf('net-epoch-%d.mat', epoch));
opts.labelingDir = fullfile(opts.expDir, sprintf('labelings-%s-epoch-%d', subset, epoch));
mapOutputFolder = fullfile(opts.expDir, sprintf('outputMaps-epoch-%d', epoch));
mappingPath = fullfile(opts.expDir, sprintf('mapping-%s.mat', subset));
resPath = fullfile(opts.expDir, sprintf('results-%s-epoch-%d.mat', subset, epoch));

% experiment setup
opts.gpus = gpus;
opts.modelFamily = modelFamily;
opts.modelType = modelType;

% Fix randomness
rng(randSeed);

% Early abort if we already know the result
if exist(resPath, 'file') && doCache
    info = load(resPath);
    return;
end

% Check if directory exists
if ~exist(opts.expDir, 'dir')
    error('Error: Experiment directory does not exist %s\n', opts.expDir);
end;

% Create dirs
if ~exist(opts.labelingDir, 'dir')
    mkdir(opts.labelingDir);
end
if doOutputMaps && ~exist(mapOutputFolder, 'dir')
    mkdir(mapOutputFolder)
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% Load imdb from file
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath);
    assert(isequal(dataset.name, imdb.dataset.name));
else
    error('Error: Cannot find imdb: %s', opts.imdbPath);
end

% Set batch options
bopts.imageNameToLabelMap = imdb.imageNameToLabelMap;
bopts.translateLabels = imdb.translateLabels;

% Get train and val sets
trn = find(imdb.images.set == 1 & imdb.images.segmentation);
val = find(imdb.images.set == 2 & imdb.images.segmentation);

% Set target dataset
if strcmp(subset, 'trn')
    target = trn;
elseif strcmp(subset, 'tst')
    target = val;
else
    error('Error: Unknown subset: %s', subset);
end
targetCount = numel(target);

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
    case 'matconvnet'
        net = load(opts.modelPath);
        
        %%% Compatibility mode for old layer names
        relIdx = find(strcmp({net.net.layers.type}, 'SegmentationAccuracyFlexible'));
        if ~isnan(relIdx)
            net.net.layers(relIdx).type = 'dagnn.SegmentationAccuracyFlexible';
        end
        
        relIdx = find(strcmp({net.net.layers.type}, 'dagnn.SegmentationLossWeighted'));
        if ~isnan(relIdx)
            net.net.layers(relIdx).type = 'dagnn.SegmentationLossPixel';
        end
        
        net = dagnn.DagNN.loadobj(net.net);
        net.mode = 'test';
        for name = {'objective', 'accuracy'}
            % Depending on the level of supervision we might not have an
            % accuracy layert
            if ~isnan(net.getLayerIndex(name))
                net.removeLayer(name);
            end
        end
        net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3);
        predVar = net.getVarIndex('prediction');
        inputVar = 'input';
        imageNeedsToBeMultiple = true;
        
    case 'ModelZoo'
        net = dagnn.DagNN.loadobj(load(opts.modelPath));
        net.mode = 'test';
        predVar = net.getVarIndex('upscore');
        inputVar = 'data';
        imageNeedsToBeMultiple = false;
        
    case 'TVG'
        net = dagnn.DagNN.loadobj(load(opts.modelPath));
        net.mode = 'test';
        predVar = net.getVarIndex('coarse');
        inputVar = 'data';
        imageNeedsToBeMultiple = false;
end

if ~isempty(opts.gpus)
    gpuDevice(opts.gpus(1));
    net.move('gpu');
end
net.mode = 'test';

% -------------------------------------------------------------------------
% Test
% -------------------------------------------------------------------------

% Prepare stuff for visualization
labelNames = dataset.getLabelNames();
colorMapping = labelColors(imdb.labelCount);
colorMappingError = [0, 0, 0; ...    % background
    1, 0, 0; ...    % too much
    1, 1, 0; ...    % too few
    0, 1, 0; ...    % rightClass
    0, 0, 1];       % wrongClass

if findMapping
    % Special mode where we use a net from a different dataset
    labelNamesPred = getIlsvrcClsClassDescriptions()';
    labelNamesPred = lower(labelNamesPred);
    labelNamesPred = cellfun(@(x) x(1:min(10, numel(x))), labelNamesPred, 'UniformOutput', false);
    colorMappingPred = labelColors(numel(labelNamesPred));
    assert(imdb.labelCount == imdb.dataset.labelCount); %imdb.labelCount should correspond to the target dataset
    confusion = zeros(imdb.labelCount, numel(labelNamesPred));
else
    % Normal test mode
    labelNamesPred = labelNames;
    colorMappingPred = colorMapping;
    confusion = zeros(imdb.labelCount);
end

evalTimer = tic;
for i = 1 : numel(target)
    imId = target(i);
    imageName = imdb.images.name{imId};
    
    % Load an image and gt segmentation
    rgb = double(imdb.dataset.getImage(imageName)) * 255;
    anno = uint16(bopts.imageNameToLabelMap(imageName, imdb));
    
    if bopts.translateLabels,
        % Before: 255 = ignore, 0 = bkg, 1:labelCount = classes
        % After : 0 = ignore, 1 = bkg, 2:labelCount+1 = classes
        anno = mod(anno + 1, 256);
    end;
    % 0 = ignore, 1:labelCount = classes
    
    % Subtract the mean (color)
    im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage);
    
    % Workaround: Limit image size to avoid running out of RAM
    if ~isempty(maxImSize),
        maxSize = max(size(im, 1), size(im, 2));
        if maxSize > maxImSize,
            factor = maxImSize / maxSize;
            im = imresize(im, factor);
        end;
    end;
    
    % Some networks requires the image to be a multiple of 32 pixels
    if imageNeedsToBeMultiple
        sz = [size(im, 1), size(im, 2)];
        sz_ = round(sz / 32) * 32;
        im_ = imresize(im, sz_);
    else
        im_ = im;
    end
    
    if ~isempty(opts.gpus)
        im_ = gpuArray(im_);
    end
    
    % Forward image through net and get predictions
    net.eval({inputVar, im_});
    scores_ = net.vars(predVar).value;
    [~, pred_] = max(scores_, [], 3);
    pred_ = gather(pred_);
    
    if imageNeedsToBeMultiple
        pred = imresize(pred_, size(anno), 'method', 'nearest');
    else
        pred = pred_;
    end
    
    % If a folder was specified, output the predicted label maps
    if doOutputMaps
        outputMap = pred; %#ok<NASGU>
        if imdb.labelCount > 200
            scoresDownsized = []; %#ok<NASGU>
        else
            scoresDownsized = scores_; %#ok<NASGU>
        end
        outputPath = fullfile(mapOutputFolder, [imageName, '.mat']);
        save(outputPath, 'outputMap', 'scoresDownsized');
    end;
    
    % Accumulate errors
    ok = anno > 0;
    confusion = confusion + accumarray([anno(ok), pred(ok)], 1, size(confusion));
    
    % Plot example images
    if mod(i - 1, plotFreq) == 0 || i == targetCount      
        
        % Print segmentation
        if showPlot,
            figure(100);
            clf;
            displayImage(rgb/255, anno, pred, imdb);
            drawnow;
        end;
        
        % Create tiled image with image+gt+pred
        if true
            if dataset.annotation.labelOneIsBg
                skipLabelInds = 1;
            else
                skipLabelInds = [];
            end;
            
            % Create tiling
            tile = ImageTile();
            
            % Add GT image
            tile.addImage(rgb/255);
            annoIm = ind2rgb(double(anno), colorMapping);
            annoIm = imageInsertBlobLabels(annoIm, anno, labelNames, 'skipLabelInds', skipLabelInds);
            tile.addImage(annoIm);
            
            % Add prediction image
            predNoBg = pred;
            predNoBg(anno == 0) = 0;
            predIm = ind2rgb(predNoBg, colorMappingPred);
            predIm = imageInsertBlobLabels(predIm, predNoBg, labelNamesPred, 'skipLabelInds', skipLabelInds);
            tile.addImage(predIm);
            
            % Highlight differences between GT and prediction
            if ~findMapping
                errorMap = ones(size(anno));
                if imdb.dataset.annotation.labelOneIsBg
                    % Datasets where bg is 1 and void is 0 (i.e. VOC)
                    tooMuch = anno ~= pred & anno == 1 & pred >= 2;
                    tooFew  = anno ~= pred & anno >= 2 & pred == 1;
                    rightClass = anno == pred & anno >= 2 & pred >= 2;
                    wrongClass = anno ~= pred & anno >= 2 & pred >= 2;
                    errorMap(tooMuch) = 2;
                    errorMap(tooFew) = 3;
                    errorMap(rightClass) = 4;
                    errorMap(wrongClass) = 5;
                else
                    % For datasets without bg
                    rightClass = anno == pred & anno >= 1;
                    wrongClass = anno ~= pred & anno >= 1;
                    errorMap(rightClass) = 4;
                    errorMap(wrongClass) = 5;
                end
                errorIm = ind2rgb(double(errorMap), colorMappingError);
                tile.addImage(errorIm);
            end
            
            % Save segmentation
            image = tile.getTiling('totalX', numel(tile.images), 'delimiterPixels', 1, 'backgroundBlack', false);
            imPath = fullfile(opts.labelingDir, [imageName, '.png']);
            imwrite(image, imPath);
        end
    end
    
    % Print message
    if mod(i - 1, printFreq) == 0 || i == targetCount
        evalTime = toc(evalTimer);
        fprintf('Processing image %d of %d (%.2f Hz)...\n', i, targetCount, i/evalTime);
    end
end


if findMapping
    % Save mapping to disk
    save(mappingPath, 'confusion');
else
    % Final statistics, remove classes missing in test
    % Note: Printing statistics earlier does not make sense if we remove missing
    % classes
    [info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion);
    info.confusion = confusion;
    fprintf('Result with all classes:\n');
    fprintf('IU %4.1f ', 100 * info.iu);
    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
        100*info.miu, 100*info.pacc, 100*info.macc);
    
    % Save results
    if doCache
        save(resPath, '-struct', 'info');
    end
end

% -------------------------------------------------------------------------
function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
% -------------------------------------------------------------------------
pos = sum(confusion, 2);
res = sum(confusion, 1)';
tp = diag(confusion);
IU = tp ./ max(1, pos + res - tp);
missing = pos == 0;
meanIU = mean(IU(~missing));
pixelAccuracy = sum(tp) / max(1, sum(confusion(:)));
meanAccuracy = mean(tp(~missing) ./ pos(~missing));

% -------------------------------------------------------------------------
function displayImage(im, lb, pred, imdb)
% -------------------------------------------------------------------------
subplot(2,2,1);
image(im);
axis image;
title('source image');

subplot(2,2,2);
image(uint8(lb-1));
axis image;
title('ground truth')

cmap = labelColors(imdb.labelCount);
subplot(2,2,3);
image(uint8(pred-1));
axis image;
title('predicted');

colormap(cmap);

% -------------------------------------------------------------------------
function cmap = labelColors(labelCount)
% -------------------------------------------------------------------------
N=labelCount;
cmap = zeros(N,3);
for i=1:N
    id = i-1; r=0;g=0;b=0;
    for j=0:7
        r = bitor(r, bitshift(bitget(id,1),7 - j));
        g = bitor(g, bitshift(bitget(id,2),7 - j));
        b = bitor(b, bitshift(bitget(id,3),7 - j));
        id = bitshift(id,-3);
    end
    cmap(i,1)=r; cmap(i,2)=g; cmap(i,3)=b;
end
cmap = cmap / 255;