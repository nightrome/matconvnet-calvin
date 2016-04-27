function[info] = fcnTestGeneric(varargin)
% [info] = fcnTestGeneric(varargin)
%
% Copyright by Holger Caesar, 2016

% Initial settings
p = inputParser;
addParameter(p, 'dataset', VOC2011Dataset());
addParameter(p, 'modelType', 'fcn8s');
addParameter(p, 'gpus', 1);
addParameter(p, 'randSeed', 42);
addParameter(p, 'expNameAppend', 'test');
addParameter(p, 'epoch', 50);
addParameter(p, 'modelFamily', 'matconvnet');
addParameter(p, 'plotFreq', 30);
addParameter(p, 'showPlot', false);
addParameter(p, 'maxImSize', 700);
addParameter(p, 'mapOutputFolder', ''); % Optional: output predicted labels to file
addParameter(p, 'doCache', true);
parse(p, varargin{:});

dataset = p.Results.dataset;
modelType = p.Results.modelType;
gpus = p.Results.gpus;
randSeed = p.Results.randSeed;
expNameAppend = p.Results.expNameAppend;
epoch = p.Results.epoch;
modelFamily = p.Results.modelFamily;
plotFreq = p.Results.plotFreq;
maxImSize = p.Results.maxImSize;
showPlot = p.Results.showPlot;
mapOutputFolder = p.Results.mapOutputFolder;
doCache = p.Results.doCache;
callArgs = p.Results; %#ok<NASGU>

% experiment and data paths
global glFeaturesFolder;
expName = [modelType, prependNotEmpty(expNameAppend, '-')];
opts.expDir = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', dataset.name, expName);
opts.modelPath = fullfile(opts.expDir, sprintf('net-epoch-%d.mat', epoch));
opts.labelingDir = fullfile(opts.expDir, sprintf('labelings-epoch-%d', epoch));

% experiment setup
opts.gpus = gpus;
opts.modelFamily = modelFamily;
opts.modelType = modelType;

% Fix randomness
rng(randSeed);

% Early abort if we already know the result
resPath = fullfile(opts.expDir, sprintf('results-epoch-%d.mat', epoch));
if exist(resPath, 'file') && doCache
    info = load(resPath);
    return;
end

% Check if directory exists
if ~exist(opts.expDir, 'dir')
    error('Error: Experiment directory does not exist %s\n', opts.expDir);
end;

% Create dirs
if ~exist(opts.labelingDir, 'dir'),
    mkdir(opts.labelingDir);
end

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

if strStartsWith(dataset.name, 'VOC')
    % Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
    % segmentations
    opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
    if exist(opts.imdbPath, 'file')
        imdb = load(opts.imdbPath);
        
        % Overwrite imdb dataset to avoid problems with changed paths
        assert(isequal(dataset.name, imdb.dataset.name));
        imdb.dataset = dataset;
    end
    % Get validation subset
    val = find(imdb.images.set == 2 & imdb.images.segmentation);
    
    bopts.imageNameToLabelMap = @(imageName, imdb) imread(sprintf(imdb.paths.classSegmentation, imageName));
    bopts.translateLabels = true;
else
    % Other datasets
    % Required fields: imdb.images.name, imdb.paths.image,
    % imdb.paths.classSegmentation, imdb.dataset, imdb.labelCount
    imdb.dataset = dataset;
    [imdb.classes.name, imdb.labelCount] = dataset.getLabelNames();
    imdb.images.name = dataset.getImageListTst();
    imdb.paths.image = fullfile(dataset.getImagePath(), sprintf('%%s%s', dataset.imageExt));
    
    val = 1:numel(imdb.images.name);
    
    bopts.imageNameToLabelMap = @(imageName, imdb) imdb.dataset.getImLabelMap(imageName);
    bopts.translateLabels = false;
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

switch opts.modelFamily
    case 'matconvnet'
        net = load(opts.modelPath);
        
        % Compatibility mode for former settings
        addDagNN = find(~strStartsWith({net.net.layers.type}, 'dagnn.'));
        for i = 1 : numel(addDagNN), net.net.layers(addDagNN(i)).type = ['dagnn.', net.net.layers(addDagNN(i)).type]; end;
        
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

confusion = zeros(imdb.labelCount);

for i = 1:numel(val)
    imId = val(i);
    imageName = imdb.images.name{imId};
    
    % Load an image and gt segmentation
    rgb = double(imdb.dataset.getImage(imageName)) * 255;
    %   rgb = round(imresize(rgb, imageSize));
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
    
    % Soome networks requires the image to be a multiple of 32 pixels
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
    
    net.eval({inputVar, im_});
    scores_ = gather(net.vars(predVar).value);
    [~, pred_] = max(scores_, [], 3);
    
    % DEBUG:
    if true,
        % Translate label indices if SiftFlow was trained with 34 classes
        if imdb.labelCount + 1 == net.layers(net.getLayerIndex('fc8')).block.size(4),
            [~, pred_NoBg] = max(scores_(:, :, 2:end),[],3);
            pred_ = pred_NoBg;
        end;
    end;
    
    if imageNeedsToBeMultiple
        pred = imresize(pred_, size(anno), 'method', 'nearest');
    else
        pred = pred_;
    end
    
    % If a folder was specified, output the predicted label maps
    if ~isempty(mapOutputFolder),
        outputMap = pred; %#ok<NASGU>
        outputPath = fullfile(mapOutputFolder, [imageName, '.mat']);
        save(outputPath, 'outputMap');
    end;
    
    % Accumulate errors
    ok = anno > 0;
    confusion = confusion + accumarray([anno(ok), pred(ok)], 1, [imdb.labelCount imdb.labelCount]);
    
    % Plots
    if mod(i - 1, plotFreq) == 0 || i == numel(val)        
        fprintf('Processing image %d of %d...\n', i, numel(val))
        
        % Print segmentation
        if showPlot,
            figure(100);
            clf;
            displayImage(rgb/255, anno, pred, imdb);
            drawnow;
        end;
        
        % Create tiled image with image+gt+pred
        labelNames = dataset.getLabelNames();
        if isa(dataset, 'VOC2011Dataset'),
            skipLabelInds = 1;
        else
            skipLabelInds = [];
        end;
        
        tile = ImageTile();
        tile.addImage(rgb/255);
        colorMapping = labelColors(imdb);
        annoIm = ind2rgb(double(anno), colorMapping);
        annoIm = imageInsertBlobLabels(annoIm, anno, labelNames, 'skipLabelInds', skipLabelInds);
        tile.addImage(annoIm);
        predIm = ind2rgb(pred, colorMapping);
        predIm = imageInsertBlobLabels(predIm, pred, labelNames, 'skipLabelInds', skipLabelInds);
        tile.addImage(predIm);
        image = tile.getTiling('totalX', 3, 'delimiterPixels', 1, 'backgroundBlack', false);
        
        % Save segmentation
        imPath = fullfile(opts.labelingDir, [imageName '.png']);
        imwrite(image, imPath);
    end
end

% Final statistics, remove classes missing in test
% Note: Printing statistics earlier does not make sense if we remove missing
% classes
[info.iu, info.miu, info.pacc, info.macc] = getAccuracies(confusion);
fprintf('Results without missing classes:\n');
fprintf('IU %4.1f ', 100 * info.iu);
fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
    100*info.miu, 100*info.pacc, 100*info.macc);

% Save results
if doCache
    save(resPath, '-struct', 'info');
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

cmap = labelColors(imdb);
subplot(2,2,3);
image(uint8(pred-1));
axis image;
title('predicted');

colormap(cmap);

% -------------------------------------------------------------------------
function cmap = labelColors(imdb)
% -------------------------------------------------------------------------
N=imdb.labelCount;
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