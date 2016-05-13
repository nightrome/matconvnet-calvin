classdef ImdbFCN < ImdbCalvin
    % ImdbFCN
    %
    % Standard Imdb for FCN experiments.
    %
    % Copyright by Holger Caesar, 2016
    
    properties
        % Set in constructor
        dataset
        imdb
        
        batchOpts = struct();
    end
    methods
        function obj = ImdbFCN(dataset, expDir, dataRootDir, nnOpts)
            % Call default constructor
            obj = obj@ImdbCalvin();
            obj.dataset = dataset;
            
            % Set folders and files
            obj.batchOpts.imdbPath = fullfile(expDir, 'imdb.mat');
            
            % FCN-specific
            obj.batchOpts.imageSize = [512, 512] - 128;
            obj.batchOpts.labelStride = 1;
            
            obj.batchOpts.imageFlipping = true;
            obj.batchOpts.rgbMean = single([128; 128; 128]);
            obj.batchOpts.classWeights = [];
            obj.batchOpts.imageNameToLabelMap = @(imageName) obj.dataset.getImLabelMap(imageName);
            obj.batchOpts.translateLabels = false;
            obj.batchOpts.maskThings = false;
            
            obj.batchOpts.useInvFreqWeights = false;
            
            % Dataset-specific
            obj.batchOpts.vocAdditionalSegmentations = true;
            obj.batchOpts.vocEdition = '11';
            obj.batchOpts.dataDir = fullfile(dataRootDir, obj.dataset.name);
            
            % Load VOC-style IMDB
            obj.loadImdb(nnOpts);
        end
        
        function loadImdb(obj, nnOpts)
            % loadImdb(obj, nnOpts)
            %
            % Creates or load the current VOC-style IMDB.
            
            if exist(obj.batchOpts.imdbPath, 'file')
                imdbInt = load(obj.batchOpts.imdbPath);
            else
                %%% VOC specific
                if strStartsWith(obj.dataset.name, 'VOC')
                    % Get PASCAL VOC segmentation dataset plus Berkeley's additional segmentations

                    imdbInt = vocSetup('dataDir', obj.batchOpts.dataDir, ...
                        'edition', obj.batchOpts.vocEdition, ...
                        'includeTest', false, ...
                        'includeSegmentation', true, ...
                        'includeDetection', false);
                    if obj.batchOpts.vocAdditionalSegmentations
                        imdbInt = vocSetupAdditionalSegmentations(imdbInt, 'dataDir', obj.batchOpts.dataDir);
                    end
                    
                    stats = getDatasetStatistics(imdbInt);
                    imdbInt.rgbMean = stats.rgbMean;
                    imdbInt.translateLabels = true;
                    imdbInt.imageNameToLabelMap = @(imageName) imread(sprintf(obj.imdb.paths.classSegmentation, imageName));
                else
                    %%% Other datasets
                    % Get labels and image path
                    imdbInt.classes.name = obj.dataset.getLabelNames();
                    imdbInt.paths.image = fullfile(obj.dataset.getImagePath(), sprintf('%%s%s', obj.dataset.imageExt));
                    
                    % Get trn + tst/val images
                    imageListTrn = obj.dataset.getImageListTrn();
                    imageListTst = obj.dataset.getImageListTst();
                    
                    % Remove images without labels
                    missingImageIndicesTrn = obj.dataset.getMissingImageIndices('train');
                    imageListTrn(missingImageIndicesTrn) = [];
                    % TODO: is it a good idea to remove test images?
                    % (only doing it on non-competitive EdiStuff
                    if isa(obj.dataset, 'EdiStuffDataset') || isa(obj.dataset, 'EdiStuffSubsetDataset')
                        missingImageIndicesTst = obj.dataset.getMissingImageIndices('test');
                        imageListTst(missingImageIndicesTst) = [];
                    end
                    imageCountTrn = numel(imageListTrn);
                    imageCountTst = numel(imageListTst);
                    
                    imdbInt.images.name = [imageListTrn; imageListTst];
                    imdbInt.images.segmentation = true(imageCountTrn+imageCountTst, 1);
                    imdbInt.images.set = nan(imageCountTrn+imageCountTst, 1);
                    imdbInt.images.set(1:imageCountTrn) = 1;
                    imdbInt.images.set(imageCountTrn+1:end) = 2;
                    
                    imdbInt.rgbMean = obj.dataset.getMeanColor();
                    imdbInt.translateLabels = false;
                    imdbInt.imageNameToLabelMap = @(imageName) obj.dataset.getImLabelMap(imageName);
                end
                
                % Dataset-independent imdb fields
                imdbInt.labelCount = obj.dataset.labelCount;
                
                % Specify level of supervision for each train image
                if ~nnOpts.misc.weaklySupervised
                    % FS
                    imdbInt.images.isFullySupervised = true(numel(imdbInt.images.name), 1);
                elseif ~semiSupervised
                    % WS
                    imdbInt.images.isFullySupervised = false(numel(imdbInt.images.name), 1);
                else
                    % SS: Set x% of train and all val to true
                    imdbInt.images.isFullySupervised = true(numel(imdbInt.images.name), 1);
                    if isa(obj.dataset, 'EdiStuffDataset')
                        selWS = find(~ismember(imdbInt.images.name, obj.dataset.datasetFS.getImageListTrn()));
                        assert(numel(selWS) == 18431);
                    else
                        selTrain = find(imdbInt.images.set == 1);
                        selTrain = selTrain(randperm(numel(selTrain)));
                        selWS = selTrain((selTrain / numel(selTrain)) >= semiSupervisedRate);
                    end
                    imdbInt.images.isFullySupervised(selWS) = false;
                    
                    if nnOpts.misc.semiSupervisedOnlyFS
                        % Keep x% of train and all val
                        selFS = imdbInt.images.isFullySupervised(:) | imdbInt.images.set(:) == 2;
                        imdbInt.images.name = imdbInt.images.name(selFS);
                        imdbInt.images.set = imdbInt.images.set(selFS);
                        imdbInt.images.segmentation = imdbInt.images.segmentation(selFS);
                        imdbInt.images.isFullySupervised = imdbInt.images.isFullySupervised(selFS);
                        
                        if strStartsWith(obj.dataset.name, 'VOC')
                            imdbInt.images.id = imdbInt.images.id(selFS);
                            imdbInt.images.classification = imdbInt.images.classification(selFS);
                            imdbInt.images.size = imdbInt.images.size(:, selFS);
                        end
                    end
                end
                
                % Make sure val images are always fully supervised
                imdbInt.images.isFullySupervised(imdbInt.images.set == 2) = true;
                
                % Print overview of the fully and weakly supervised number of training
                % images
                fsCount = sum( imdbInt.images.isFullySupervised(:) & imdbInt.images.set(:) == 1);
                wsCount = sum(~imdbInt.images.isFullySupervised(:) & imdbInt.images.set(:) == 1);
                fsRatio = fsCount / (fsCount+wsCount);
                wsRatio = 1 - fsRatio;
                fprintf('Images in train: %d FS (%.1f%%), %d WS (%.1f%%)...\n', fsCount, fsRatio * 100, wsCount, wsRatio * 100);
                
                % Save imdb
                save(obj.batchOpts.imdbPath, '-struct', 'imdbInt');
            end
            
            % Get training and test/validation subsets
            % We always validate and test on val
            obj.data.train = find(imdbInt.images.set == 1 & imdbInt.images.segmentation);
            obj.data.val = find(imdbInt.images.set == 2 & imdbInt.images.segmentation);
            
            % Store in class object
            obj.imdb = imdbInt;
        end
        
        function[inputs, numElements] = getBatch(obj, batchIdx, net, nnOpts)
            % [inputs, numElements] = getBatch(obj, batchIdx, net, nnOpts)
            %
            % Returns a batch ...
            % Note: Currently train and val batches are treated the same.
            %
            % Copyright by Holger Caesar, 2015
            
            % Check inputs
            assert(~isempty(obj.datasetMode));
            assert(numel(batchIdx) == 1);
            
            % Dummy init
            inputs = {};
            numElements = 0;
            
            % Determine whether testing
            testMode = strcmp(obj.datasetMode, 'test');
            
            %%%
            
            % Check settings
            assert(~isempty(obj.batchOpts.rgbMean));
            obj.batchOpts.rgbMean = reshape(obj.batchOpts.rgbMean, [1 1 3]);
            
            % Make sure that the subbatch size is one image
            imageCount = numel(batchIdx);
            if imageCount == 0
                % Empty batch
                return;
            elseif imageCount == 1
                % Default
            else
                error('Error: GetBatch cannot process more than 1 image at a time!');
            end
            imageIdx = batchIdx;
            
            % Initializations
            image = zeros(obj.batchOpts.imageSize(1), obj.batchOpts.imageSize(2), 3, imageCount, 'single');
            lx = 1 : obj.batchOpts.labelStride : obj.batchOpts.imageSize(2);
            ly = 1 : obj.batchOpts.labelStride : obj.batchOpts.imageSize(1);
            labels = zeros(numel(ly), numel(lx), 1, imageCount, 'double'); % must be double for to avoid numerical precision errors in vl_nnloss, when using many classes
            if nnOpts.misc.weaklySupervised
                labelsImageCell = cell(imageCount, 1);
            end
            if nnOpts.misc.maskThings
                assert(isa(obj.dataset, 'EdiStuffDataset'));
                datasetIN = ImageNetDataset();
            end
            masksThingsCell = cell(imageCount, 1); % by default this is empty
            
            if true
                % Get image
                imageName = obj.imdb.images.name{imageIdx};
                rgb = double(obj.dataset.getImage(imageName)) * 255;
                if size(rgb,3) == 1
                    rgb = cat(3, rgb, rgb, rgb);
                end
                
                % Get pixel-level GT
                if obj.dataset.annotation.hasPixelLabels || obj.imdb.images.isFullySupervised(imageIdx)
                    anno = uint16(obj.batchOpts.imageNameToLabelMap(imageName));
                    
                    % Translate labels s.t. 255 is mapped to 0
                    if obj.batchOpts.translateLabels,
                        % Before: 255 = ignore, 0 = bkg, 1:labelCount = classes
                        % After : 0 = ignore, 1 = bkg, 2:labelCount+1 = classes
                        anno = mod(anno + 1, 256);
                    end
                    % 0 = ignore, 1:labelCount = classes
                else
                    anno = [];
                end
                
                % Crop and rescale image
                h = size(rgb, 1);
                w = size(rgb, 2);
                sz = obj.batchOpts.imageSize(1 : 2);
                scale = max(h / sz(1), w / sz(2));
                scale = scale .* (1 + (rand(1) - .5) / 5);
                sy = round(scale * ((1:sz(1)) - sz(1)/2) + h/2);
                sx = round(scale * ((1:sz(2)) - sz(2)/2) + w/2);
                
                % Flip image
                if obj.batchOpts.imageFlipping && rand > 0.5
                    sx = fliplr(sx);
                end
                
                % Get image indices in valid area
                okx = find(1 <= sx & sx <= w);
                oky = find(1 <= sy & sy <= h);
                
                % Subtract mean image
                image(oky, okx, :, 1) = bsxfun(@minus, rgb(sy(oky), sx(okx), :), obj.batchOpts.rgbMean);
                
                % Fully supervised: Get pixel level labels
                if ~isempty(anno)
                    tlabels = zeros(sz(1), sz(2), 'double');
                    tlabels(oky,okx) = anno(sy(oky), sx(okx));
                    tlabels = single(tlabels(ly,lx));
                    labels(:, :, 1, 1) = tlabels; % 0: ignore
                end
                
                % Weakly supervised: extract image-level labels
                if nnOpts.misc.weaklySupervised
                    if ~isempty(anno) && ~all(anno(:) == 0)
                        % Get image labels from pixel labels
                        % These are already translated (if necessary)
                        curLabelsImage = unique(anno);
                    else
                        curLabelsImage = obj.dataset.getImLabelInds(imageName);
                        
                        % Translate labels s.t. 255 is mapped to 0
                        if obj.batchOpts.translateLabels
                            curLabelsImage = mod(curLabelsImage + 1, 256);
                        end
                        
                        if obj.dataset.annotation.labelOneIsBg
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
                if nnOpts.misc.maskThings
                    % Get mask
                    longName = datasetIN.shortNameToLongName(imageName);
                    mask = datasetIN.getImLabelBoxesMask(longName);
                    
                    % Resize it if necessary
                    if size(mask, 1) ~= size(image, 1) || ...
                            size(mask, 2) ~= size(image, 2)
                        mask = imresize(mask, [size(image, 1), size(image, 2)]);
                    end
                    masksThingsCell{1} = mask;
                end
            end
            
            % Extract inverse class frequencies from dataset
            if obj.batchOpts.useInvFreqWeights,
                if nnOpts.misc.weaklySupervised,
                    classWeights = obj.dataset.getLabelImFreqs('train');
                else
                    classWeights = obj.dataset.getLabelPixelFreqs('train');
                end
                
                % Inv freq and normalize classWeights
                classWeights = classWeights ./ sum(classWeights);
                nonEmpty = classWeights ~= 0;
                classWeights(nonEmpty) = 1 ./ classWeights(nonEmpty);
                classWeights = classWeights ./ sum(classWeights);
                assert(~any(isnan(classWeights)));
            else
                classWeights = [];
            end
            obj.batchOpts.classWeights = classWeights;
            
            % Move image to GPU
            if strcmp(net.device, 'gpu')
                image = gpuArray(image);
            end
            
            % Store in output struct
            inputs = {'input', image};
            if obj.dataset.annotation.hasPixelLabels || obj.imdb.images.isFullySupervised(imageIdx)
                inputs = [inputs, {'label', labels}];
            end
            if nnOpts.misc.weaklySupervised
                assert(~any(cellfun(@(x) isempty(x), labelsImageCell)));
                inputs = [inputs, {'labelsImage', labelsImageCell}];
            end
            
            % Instance/pixel weights, can be left empty
            inputs = [inputs, {'classWeights', obj.batchOpts.classWeights}];
            
            % Decide which level of supervision to pick
            if nnOpts.misc.semiSupervised
                % SS
                isWeaklySupervised = ~obj.imdb.images.isFullySupervised(batchIdx);
            else
                % FS or WS
                isWeaklySupervised = nnOpts.misc.weaklySupervised;
            end
            inputs = [inputs, {'isWeaklySupervised', isWeaklySupervised}];
            inputs = [inputs, {'masksThingsCell', masksThingsCell}];
            numElements = 1; % One image
        end
        
        function[allBatchInds] = getAllBatchInds(obj)
            % Obtain the indices and ordering of all batches (for this epoch)
            
            batchCount = size(obj.data.(obj.datasetMode), 1);
            if strcmp(obj.datasetMode, 'train')
                allBatchInds = randperm(batchCount);
            elseif strcmp(obj.datasetMode, 'val'),
                allBatchInds = 1:batchCount;
            elseif strcmp(obj.datasetMode, 'test'),
                allBatchInds = 1:batchCount;
            else
                error('Error: Unknown datasetMode!');
            end
        end
        
        function initEpoch(obj, epoch)
            % Call default method
            initEpoch@ImdbCalvin(obj, epoch);
        end
    end
end