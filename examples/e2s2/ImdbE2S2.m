classdef ImdbE2S2 < ImdbCalvin
    % ImdbE2S2
    %
    % Standard Imdb for all fully and weakly supervised E2S2 experiments.
    %
    % Copyright by Holger Caesar, 2016
    
    properties
        % Set in constructor
        dataset
        segmentFolder
        
        % Automatically set
        segmentFolderRP
        segmentFolderSP
        segmentFolderGT
        
        batchOpts = struct();
        imageSample = struct();
    end
    methods
        function obj = ImdbE2S2(dataset, segmentFolder)
            % Call default constructor
            obj = obj@ImdbCalvin();
            
            % Set default options
            obj.batchOpts.maxImageSize = 600;
            obj.batchOpts.negRange = [0.0, 0.1];
            obj.batchOpts.posRange = [0.1, 1.0];
            obj.batchOpts.subsample = true;
            obj.batchOpts.removeGT = false;
            obj.batchOpts.overlapThreshGTSP = 0.5;
            obj.batchOpts.blobMaxSize = []; % relative to image size
            obj.batchOpts.imageFlipping = true;
            obj.batchOpts.segments.minSize = 100;
            obj.batchOpts.segments.colorTypeIdx = 1; % should always be 1 on start
            obj.batchOpts.segments.switchColorTypesEpoch = false;
            obj.batchOpts.segments.switchColorTypesBatch = false;
            obj.batchOpts.segments.colorTypes = {'Rgb'};
            obj.batchOpts.segments.segmentStrRP = 'Uijlings2013-ks%d-sigma0.8-colorTypes%s';
            obj.batchOpts.segments.segmentStrSP = 'Felzenszwalb2004-k%d-sigma0.8-colorTypes%s';
            obj.batchOpts.segments.segmentStrGT = 'GroundTruth';
            obj.imageSample.use = false;
            
            % Set segment names
            obj.dataset = dataset;
            obj.segmentFolder = segmentFolder;
            obj.updateSegmentNames();
            
            % Reset global variables
            global labelPixelFreqsOriginal;
            labelPixelFreqsOriginal = [];
        end
        
        function[inputs, numElements] = getBatch(obj, batchIdx, net, nnOpts)
            % [inputs, numElements] = getBatch(obj, batchIdx, net, nnOpts)
            %
            % Returns a batch consisting of positive and negative samples for one
            % image. The labels are empty, as the regiontopixel layer will replace
            % them.
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
            
            % Switch color type if specified
            % (this has to happen on batchOpts, not batchOptsCopy!)
            if obj.batchOpts.segments.switchColorTypesBatch && ~testMode,
                obj.switchColorType();
            end;
            
            % Create a copy of batchOpts to avoid changes to the entire imdb
            batchOptsCopy = obj.batchOpts;
            
            % Special settings for test mode (not val)
            if testMode,
                batchOptsCopy.subsample = false;
                batchOptsCopy.removeGT = true;
                batchOptsCopy.blobMaxSize = [];
                batchOptsCopy.imageFlipping = false;
                batchOptsCopy.segments.switchColorTypesEpoch = false;
                batchOptsCopy.segments.switchColorTypesBatch = false;
                if batchOptsCopy.segments.colorTypeIdx == 1,
                    batchOptsCopy.segments.colorTypeIdx = 1;
                    obj.updateSegmentNames(batchOptsCopy);
                end;
                
                if testMode && isfield(nnOpts.misc, 'testOpts'),
                    if isfield(nnOpts.misc.testOpts, 'subsamplePosRange'),
                        batchOptsCopy.subsample = true;
                        batchOptsCopy.posRange = nnOpts.misc.testOpts.subsamplePosRange;
                    end
                end;
            end;
            
            % Get params from layers and nnOpts
            roiPool = nnOpts.misc.roiPool;
            roiPool.size = net.layers(net.getLayerIndex('roipool5')).block.poolSize;
            regionToPixel = nnOpts.misc.regionToPixel;
            if isfield(nnOpts.misc, 'weaklySupervised'),
                weaklySupervised = nnOpts.misc.weaklySupervised;
                
                if weaklySupervised.use,
                    batchOptsCopy.subsample = false;
                    batchOptsCopy.removeGT = true;
                end;
            else
                weaklySupervised.use = false;
            end;
            
            % Load image
            imageIdx = batchIdx;
            imageName = obj.data.(obj.datasetMode){imageIdx};
            image = single(obj.dataset.getImage(imageName));
            
            % Move image to GPU
            if strcmp(net.device, 'gpu'),
                image = gpuArray(image);
            end;
            
            % Resize image and subtract mean image
            [image, oriImSize] = e2s2_prepareImage(net, image, batchOptsCopy.maxImageSize);
            
            % Get segmentation structure
            segmentPathRP = [obj.segmentFolderRP, filesep, imageName, '.mat'];
            segmentStructRP = load(segmentPathRP, 'propBlobs', 'overlapList', 'superPixelInds', 'superPixelLabelHistos');
            overlapListRP = segmentStructRP.overlapList;
            spInds = segmentStructRP.superPixelInds;
            spLabelHistos = segmentStructRP.superPixelLabelHistos;
            blobsRP = segmentStructRP.propBlobs(:);
            clearvars segmentStructRP;
            
            % Get blobMasks from file
            if roiPool.freeform.use,
                blobMasksName = sprintf('blobMasks%dx%d', roiPool.size(1), roiPool.size(2));
                segmentStructRP = load(segmentPathRP, blobMasksName);
                if ~isfield(segmentStructRP, blobMasksName),
                    error('Error: Missing blob masks, please run e2s2_storeBlobMasks()!');
                end;
                blobMasksRP = segmentStructRP.(blobMasksName);
                clearvars segmentStructRP;
            end;
            
            % Get superpixels
            blobsSP = blobsRP(spInds);
            clearvars spInds;
            
            if ~weaklySupervised.use
                % Get GT structure
                segmentPathGT = [obj.segmentFolderGT, filesep, imageName, '.mat'];
                segmentStructGT = load(segmentPathGT, 'propBlobs', 'labelListGT');
                blobsGT = segmentStructGT.propBlobs(:);
                labelListGT = segmentStructGT.labelListGT;
                if isempty(blobsGT),
                    % Skip images without GT regions
                    return;
                end;
                
                % Get blobMasks from file
                if roiPool.freeform.use,
                    segmentStructGT = load(segmentPathGT, blobMasksName);
                    blobMasksGT = segmentStructGT.(blobMasksName);
                end;
            end
            
            % Filter blobs according to IOU with GT
            if batchOptsCopy.subsample && ~weaklySupervised.use,
                % Compute IOUs between RP and GT
                overlapRPGT = scoreBlobIoUs(blobsRP, blobsGT);
                
                % Compute IOUs between RP and each label
                blobCountRP = numel(blobsRP);
                overlapRPLabels = zeros(blobCountRP, obj.numClasses);
                for labelIdx = 1 : obj.numClasses,
                    sel = labelListGT == labelIdx;
                    
                    if any(sel),
                        overlapRPLabels(:, labelIdx) = max(overlapRPGT(:, sel), [], 2);
                    end;
                end;
                
                % Compute maximum overlap and labels
                [maxOverlap, ~] = max(overlapRPLabels, [], 2);
                
                % Find positives (negatives are rejected)
                blobIndsRP = find(batchOptsCopy.posRange(1) <= maxOverlap & maxOverlap <= batchOptsCopy.posRange(2));
            else
                blobIndsRP = (1:numel(blobsRP))';
            end;
            
            % Remove very big blobs
            if isfield(batchOptsCopy, 'blobMaxSize') && ~isempty(batchOptsCopy.blobMaxSize) && batchOptsCopy.blobMaxSize ~= 1,
                imagePixelSize = oriImSize(1) * oriImSize(2);
                pixelSizesRP = [blobsRP.size]';
                blobIndsRP = intersect(blobIndsRP, find(pixelSizesRP <= imagePixelSize * batchOptsCopy.blobMaxSize));
            end;
            
            % Additional testing options (limit regions etc.)
            if testMode && isfield(nnOpts.misc, 'testOpts'),
                % Default arguments
                testOpts = nnOpts.misc.testOpts;
                if ~isfield(testOpts, 'maxSizeRel') || isempty(testOpts.maxSizeRel),
                    maxSizeRel = 1;
                else
                    maxSizeRel = testOpts.maxSizeRel;
                end;
                if ~isfield(testOpts, 'minSize') || isempty(testOpts.minSize),
                    minSize = 0;
                else
                    minSize = testOpts.minSize;
                end;
                
                % Select regions
                imagePixelSize = oriImSize(1) * oriImSize(2);
                pixelSizesRP = [blobsRP.size]';
                regionSel = pixelSizesRP >= minSize & pixelSizesRP / imagePixelSize <= maxSizeRel;
                blobIndsRP = intersect(blobIndsRP, find(regionSel));
            end;
            
            % At test time, make sure the whole image is included
            % (otherwise superpixels might be unlabeled!)
            % (this obviously limits the effect of maxSizeRel)
            if testMode,
                wholeImageRegion = find([blobsRP.size] == oriImSize(1) * oriImSize(2), 1);
                blobIndsRP = union(blobIndsRP, wholeImageRegion);
            end;
            
            % Apply selection to relevant fields
            blobsRP = blobsRP(blobIndsRP);
            overlapListRP = overlapListRP(blobIndsRP, :);
            if roiPool.freeform.use,
                blobMasksRP = blobMasksRP(blobIndsRP);
            end;
            
            % Compute pixel-level label frequencies (also used without inv-freqs)
            if regionToPixel.use && ~weaklySupervised.use,
                global labelPixelFreqsOriginal; %#ok<TLEV>
                if isempty(labelPixelFreqsOriginal),
                    [labelPixelFreqsSum, labelPixelImageCount] = obj.dataset.getLabelPixelFreqs();
                    labelPixelFreqsOriginal = labelPixelFreqsSum ./ labelPixelImageCount;
                end;
                
                % Cutoff extreme frequencies if required
                % Normalizes the sum to one
                if isfield(regionToPixel, 'minPixFreq') && ~isempty(regionToPixel.minPixFreq),
                    labelPixelFreqs = freqClampMinimum(labelPixelFreqsOriginal, regionToPixel.minPixFreq);
                else
                    labelPixelFreqs = labelPixelFreqsOriginal;
                end;
            end;
            
            % Merge RP and GT
            blobsAll = blobsRP;
            overlapListAll = overlapListRP;
            if roiPool.freeform.use,
                blobMasksAll = blobMasksRP;
            end;
            
            if ~batchOptsCopy.removeGT,
                % Figure out which superpixels are part of a GT region and
                % remove GT regions that don't overlap enough with a superpixel
                % Note: the overlaps are precomputed for speedup
                pixelSizesSP = [blobsSP.size]';
                segmentPathSP = [obj.segmentFolderSP, filesep, imageName, '.mat'];
                if exist(segmentPathSP, 'file'),
                    segmentStructSP = load(segmentPathSP, 'overlapRatiosSPGT');
                    overlapRatiosSPGT = segmentStructSP.overlapRatiosSPGT;
                else
                    imagePixelSize = oriImSize(1) * oriImSize(2);
                    overlapRatiosSPGT = computeBlobOverlapSum(blobsSP, blobsGT, imagePixelSize);
                end;
                overlapRatiosSPGT = bsxfun(@rdivide, overlapRatiosSPGT, pixelSizesSP);
                overlapListGT = sparse(overlapRatiosSPGT' >= batchOptsCopy.overlapThreshGTSP);
                overlappingGT = full(sum(overlapListGT, 2) > 0);
                
                % Apply selection to GT
                blobsGT = blobsGT(overlappingGT);
                overlapListGT = overlapListGT(overlappingGT, :);
                if roiPool.freeform.use,
                    blobMasksGT = blobMasksGT(overlappingGT);
                end;
                
                % Merge RP and GT
                blobsAll = [blobsAll; blobsGT];
                overlapListAll = [overlapListAll; overlapListGT];
                if roiPool.freeform.use,
                    blobMasksAll = [blobMasksAll; blobMasksGT];
                    assert(numel(blobMasksAll) == size(blobsAll, 1));
                end;
            end;
            assert(size(blobsAll, 1) == size(overlapListAll, 1));
            
            % Skip images without blobs (no RP and no GT after filtering)
            if isempty(blobsAll),
                return;
            end;
            
            % Create boxes at the very end to avoid inconsistency
            boxesAll = single(cell2mat({blobsAll.rect}'));
            assert(size(blobsAll, 1) == size(boxesAll, 1));
            
            % Store regionToPixel info in a struct
            if regionToPixel.use,
                regionToPixelAux.overlapListAll = overlapListAll;
                if ~testMode && ~weaklySupervised.use,
                    assert(size(spLabelHistos, 1) == numel(blobsSP));
                    regionToPixelAux.labelPixelFreqs = labelPixelFreqs;
                    regionToPixelAux.spLabelHistos = spLabelHistos;
                end;
            end;
            
            % Flip image, boxes and masks
            if batchOptsCopy.imageFlipping && rand() >= 0.5,
                if roiPool.freeform.use,
                    [image, boxesAll, blobMasksAll] = e2s2_flipImageBoxes(image, boxesAll, oriImSize, blobMasksAll);
                else
                    [image, boxesAll] = e2s2_flipImageBoxes(image, boxesAll, oriImSize);
                end;
            end;
            
            if weaklySupervised.use && ~testMode,
                % Get the image-level labels and weights
                % Note: these should be as similar as possible to the ones
                % in the regiontopixel layer.
                [labelNames, labelCount] = obj.dataset.getLabelNames();
                imLabelNames = obj.dataset.getImLabelList(imageName);
                if isempty(imLabelNames),
                    % Skip images that have no labels
                    return;
                end;
                imLabelInds = find(ismember(labelNames, imLabelNames));
                
                % Determine image-level frequencies
                % Assume all labels take the same number of pixels in that image
                % Note: On average it should be: sum(imLabelWeights) = 1
                % TODO: the labels are not weighted by superpixel size yet
                %       would that even make sense?
                if weaklySupervised.invLabelFreqs,
                    [labelImFreqs, ~] = obj.dataset.getLabelImFreqs();
                    labelImFreqsNorm = labelImFreqs / sum(labelImFreqs) * labelCount;
                    imLabelWeights = 1 ./ labelImFreqsNorm(imLabelInds);
                    
                    % Renormalize mass to (average of) 1
                    if weaklySupervised.normalizeImageMass,
                        imLabelWeights = imLabelWeights ./ sum(imLabelWeights);
                    end;
                else
                    imLabelCount = numel(imLabelInds);
                    imLabelWeights = repmat(1 / imLabelCount, [imLabelCount, 1]);
                end;
                
                % Reshape to loss layer format
                imLabelInds    = reshape(imLabelInds, 1, 1, 1, []);
                imLabelWeights = reshape(imLabelWeights, 1, 1, 1, []);
                assert(numel(imLabelInds) == numel(imLabelWeights));
                assert(~any(imLabelWeights == 0)); %temporary check for bugs, to be removed
            end;
            
            % Convert boxes to transposed Girshick format
            boxesAll = boxesAll(:, [2, 1, 4, 3])';
            
            % Store in output struct
            inputs = {'input', image, 'oriImSize', oriImSize, 'boxes', boxesAll};
            if regionToPixel.use,
                inputs = [inputs, {'regionToPixelAux', regionToPixelAux}];
            end;
            if ~testMode,
                inputs = [inputs, {'label', []}];
            end;
            if roiPool.freeform.use,
                inputs = [inputs, {'blobMasks', blobMasksAll}];
            end;
            if weaklySupervised.use  ...
                    && weaklySupervised.labelPresence.use ...
                    && ~testMode,
                inputs = [inputs, {'labelImage', imLabelInds}];
                inputs = [inputs, {'weightsImage', imLabelWeights}];
            end;
            numElements = 1; % One image
        end
        
        function[allBatchInds] = getAllBatchInds(obj)
            % Obtain the indices and ordering of all batches (for this epoch)
            
            batchCount = size(obj.data.(obj.datasetMode), 1);
            if strcmp(obj.datasetMode, 'train'),
                if obj.imageSample.use,
                    allBatchInds = obj.imageSample.func.(obj.datasetMode)(batchCount);
                else
                    allBatchInds = randperm(batchCount);
                end;
            elseif strcmp(obj.datasetMode, 'val'),
                if obj.imageSample.use,
                    allBatchInds = obj.imageSample.func.(obj.datasetMode)(batchCount);
                else
                    allBatchInds = 1:batchCount;
                end;
            elseif strcmp(obj.datasetMode, 'test'),
                allBatchInds = 1:batchCount;
            else
                error('Error: Unknown datasetMode!');
            end;
        end
        
        function switchColorType(obj)
            % switchColorType(obj)
            %
            % Switch to the next color type (if more than 1).
            
            obj.batchOpts.segments.colorTypeIdx = 1 + mod(obj.batchOpts.segments.colorTypeIdx, numel(obj.batchOpts.segments.colorTypes));
            obj.updateSegmentNames();
        end
        
        function updateSegmentNames(obj, batchOpts)
            % updateSegmentNames(obj, [batchOpts])
            %
            % Update the names of the current proposals/superpixels,
            % matching the current colorType and minSize.
            
            if ~exist('batchOpts', 'var'),
                batchOpts = obj.batchOpts;
            end;
            
            colorType = batchOpts.segments.colorTypes{batchOpts.segments.colorTypeIdx};
            minSize = batchOpts.segments.minSize;
            obj.segmentFolderRP = fullfile(obj.segmentFolder, sprintf(obj.batchOpts.segments.segmentStrRP, minSize, colorType));
            obj.segmentFolderSP = fullfile(obj.segmentFolder, sprintf(obj.batchOpts.segments.segmentStrSP, minSize, colorType));
            obj.segmentFolderGT = fullfile(obj.segmentFolder, obj.batchOpts.segments.segmentStrGT);
        end
        
        function initEpoch(obj, epoch)
            % Call default method
            initEpoch@ImdbCalvin(obj, epoch);
            
            % Change color type if option is selected
            % This is not related to LR flipping, but the only way to
            % currently implement this.
            if obj.batchOpts.segments.switchColorTypesEpoch,
                obj.switchColorType();
            end;
        end
    end
end