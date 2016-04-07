classdef E2S2NN < CalvinNN
    % E2S2NN End-to-end region based semantic segmentation training.
    %
    % Copyright by Holger Caesar, 2015
    
    methods
        function obj = E2S2NN(net, imdb, nnOpts)
            obj = obj@CalvinNN(net, imdb, nnOpts);
        end
        
        function convertNetwork(obj)
            % convertNetwork(obj)
            
            % Run default conversion method
            convertNetwork@CalvinNN(obj);
            
            % Insert a regiontopixel layer before the loss
            if obj.nnOpts.misc.regionToPixel.use,
                regionToPixelOpts = obj.nnOpts.misc.regionToPixel;
                regionToPixelOpts = rmfield(regionToPixelOpts, 'use');
                regionToPixelOpts = struct2Varargin(regionToPixelOpts);
                regionToPixelBlock = dagnn.RegionToPixel(regionToPixelOpts{:});
                obj.net.insertLayer('fc8', 'softmaxloss', 'regiontopixel8', regionToPixelBlock, 'regionToPixelAux', {'label', 'instanceWeights'});
            end;
            
            % Add batch normalization before ReLUs if specified
            if isfield(obj.nnOpts.misc, 'batchNorm') && obj.nnOpts.misc.batchNorm,
                reluInds = find(arrayfun(@(x) isa(x.block, 'dagnn.ReLU'), obj.net.layers));
                
                for i = 1 : numel(reluInds),
                    % Relu
                    reluIdx = reluInds(i);
                    reluLayerName = obj.net.layers(reluIdx).name;
                    reluInputIdx = obj.net.layers(reluIdx).inputIndexes;
                    assert(numel(reluInputIdx) == 1);
                    
                    % Left layer
                    leftLayerIdx = find(arrayfun(@(x) ismember(reluInputIdx, x.outputIndexes), obj.net.layers));
                    assert(numel(leftLayerIdx) == 1);
                    leftLayerName = obj.net.layers(leftLayerIdx).name;
                    leftParamIdx = obj.net.layers(leftLayerIdx).paramIndexes(1);
                    numChannels = size(obj.net.params(leftParamIdx).value, 4); % equals size(var, 3) of the input variable
                    
                    % Insert new layer
                    layerBlock = dagnn.BatchNorm('numChannels', numChannels);
                    layerParamValues = layerBlock.initParams();
                    layerName = sprintf('bn_%s', reluLayerName);
                    layerParamNames = cell(1, numel(layerParamValues));
                    for i = 1 : numel(layerParamValues), %#ok<FXSET>
                        layerParamNames{i} = sprintf('%s_%d', layerName, i);
                    end;
                    obj.net.insertLayer(leftLayerName, reluLayerName, layerName, layerBlock, {}, {}, layerParamNames);
                    
                    for i = 1 : numel(layerParamValues), %#ok<FXSET>
                        paramIdx = obj.net.getParamIndex(layerParamNames{i});
                        obj.net.params(paramIdx).value = layerParamValues{i};
                        obj.net.params(paramIdx).learningRate = 1; %TODO: are these good values?
                        obj.net.params(paramIdx).weightDecay = 0;
                    end;
                end;
            end;
            
            %%% Weakly supervised learning options
            % Check that only one ws option is chosen
            if isfield(obj.nnOpts.misc, 'weaklySupervised'),
                weaklySupervised = obj.nnOpts.misc.weaklySupervised;
            else
                weaklySupervised.use = false;
            end;
            
            if weaklySupervised.use,
                % Incur a loss per class
                if isfield(weaklySupervised, 'labelPresence') && weaklySupervised.labelPresence.use,
                    assert(obj.nnOpts.misc.regionToPixel.use);
                    
                    % Insert a labelpresence layer
                    labelPresenceOpts = weaklySupervised.labelPresence;
                    labelPresenceOpts = rmfield(labelPresenceOpts, 'use');
                    labelPresenceOpts = struct2Varargin(labelPresenceOpts);
                    labelPresenceBlock = dagnn.LabelPresence(labelPresenceOpts{:});
                    obj.net.insertLayer('regiontopixel8', 'softmaxloss', 'labelpresence', labelPresenceBlock, {}, {'labelImage'}, {});
                    labelPresenceIdx = obj.net.getLayerIndex('labelpresence');
                    obj.net.layers(labelPresenceIdx).inputs{2} = 'labelImage';
                    obj.net.layers(labelPresenceIdx).inputs(3)  = []; % Remove instanceWeights
                    obj.net.layers(labelPresenceIdx).outputs(2) = []; % Remove labelImage
                    obj.net.rebuild();
                    
                    % Change parameters for loss
                    softmaxIdx = obj.net.getLayerIndex('softmaxloss');
                    softmaxBlock = obj.net.layers(softmaxIdx).block;
                    scoresVar = obj.net.layers(softmaxIdx).inputs{1};
                    obj.net.replaceLayer('softmaxloss', 'softmaxloss', softmaxBlock, {scoresVar, 'labelImage', 'weightsImage'}, {}, {}, true);
                end
            end;
        end
        
        function[stats] = testOnSet(obj, varargin)
            % [stats] = testOnSet(obj, varargin)
            
            % Initial settings
            p = inputParser;
            addParameter(p, 'subset', 'test');
            addParameter(p, 'computeMeanIou', true);
            addParameter(p, 'cacheProbs', true);
            addParameter(p, 'limitImageCount', Inf);
            parse(p, varargin{:});
            
            subset = p.Results.subset;
            computeMeanIou = p.Results.computeMeanIou;
            cacheProbs = p.Results.cacheProbs;
            limitImageCount = p.Results.limitImageCount;
            
            % Set the datasetMode to be active
            if strcmp(subset, 'test'),
                temp = [];
            else
                temp = obj.imdb.data.test;
                obj.imdb.data.test = obj.imdb.data.(subset);
            end
            
            % Run test
            stats = obj.test('subset', subset, 'computeMeanIou', computeMeanIou, 'cacheProbs', cacheProbs, 'limitImageCount', limitImageCount);
            if ~strcmp(subset, 'test'),
                stats.loss = [obj.stats.(subset)(end).objective]';
            end;
            
            % Restore the original test set
            if ~isempty(temp),
                obj.imdb.data.test = temp;
            end;
        end
        
        function extractFeatures(obj, featFolder)
            % extractFeatures(obj, featFolder)
            
            % Init
            imageList = unique([obj.imdb.data.train; obj.imdb.data.val; obj.imdb.data.test]);
            imageCount = numel(imageList);
            
            % Update imdb's test set
            tempTest = obj.imdb.data.test;
            obj.imdb.data.test = imageList;
            
            % Set network to testing mode
            outputVarIdx = obj.prepareNetForTest();
            
            for imageIdx = 1 : imageCount,
                printProgress('Classifying images', imageIdx, imageCount, 10);
                
                % Get batch
                inputs = obj.imdb.getBatch(imageIdx, obj.net);
                
                % Run forward pass
                obj.net.eval(inputs);
                
                % Extract probs
                curProbs = obj.net.vars(outputVarIdx).value;
                curProbs = gather(reshape(curProbs, [size(curProbs, 3), size(curProbs, 4)]))';
                
                % Store
                imageName = imageList{imageIdx};
                featPath = fullfile(featFolder, [imageName, '.mat']);
                features = double(curProbs); %#ok<NASGU>
                save(featPath, 'features', '-v6');
            end;
            
            % Reset test set
            obj.imdb.data.test = tempTest;
        end
        
        function[outputVarIdx] = prepareNetForTest(obj)
            % [outputVarIdx] = prepareNetForTest(obj)
            
            % Move to GPU
            if ~isempty(obj.nnOpts.gpus),
                obj.net.move('gpu');
            end;
            
            % Enable test mode
            obj.imdb.setDatasetMode('test');
            obj.net.mode = 'test';
            
            % Reset segments to default
            obj.imdb.batchOpts.segments.colorTypeIdx = 1;
            obj.imdb.updateSegmentNames();
            
            % Get pixel output variable name
            regiontopixelIdx = obj.net.getLayerIndex('regiontopixel8');
            regiontopixelOutput = obj.net.layers(regiontopixelIdx).outputs{1};
            
            % Disable labelpresence layer (these needs to happen before we
            % remove the softmax layer)
            labelpresenceIdx = obj.net.getLayerIndex('labelpresence');
            if ~isnan(labelpresenceIdx),
                obj.net.removeLayer('labelpresence');
                softmaxlossIdx = obj.net.getLayerIndex('softmaxloss');
                obj.net.layers(softmaxlossIdx).inputs{1} = regiontopixelOutput;
                obj.net.rebuild();
            end;
            
            % Replace softmaxloss by softmax
            lossIdx = find(cellfun(@(x) isa(x, 'dagnn.Loss'), {obj.net.layers.block}));
            lossName = obj.net.layers(lossIdx).name;
            lossType = obj.net.layers(lossIdx).block.loss;
            lossInputs = obj.net.layers(lossIdx).inputs;
            if strcmp(lossType, 'softmaxlog'),
                obj.net.removeLayer(lossName);
                outputLayerName = 'softmax';
                obj.net.addLayer(outputLayerName, dagnn.SoftMax(), lossInputs{1}, 'scores', {});
                outputLayerIdx = obj.net.getLayerIndex(outputLayerName);
                outputVarIdx = obj.net.layers(outputLayerIdx).outputIndexes;
            elseif strcmp(lossType, 'log'),
                % Only output the scores of the regiontopixel layer
                obj.net.removeLayer(lossName);
                outputVarIdx = obj.net.getVarIndex(obj.net.getOutputs{1});
            else
                error('Error: Unknown loss function!');
            end;
            assert(numel(outputVarIdx) == 1);
        end
        
        function[stats] = test(obj, varargin)
            % [stats] = test(obj, varargin)
            
            % Initial settings
            p = inputParser;
            addParameter(p, 'subset', 'test');
            addParameter(p, 'computeMeanIou', true);
            addParameter(p, 'cacheProbs', true);
            addParameter(p, 'limitImageCount', Inf);
            parse(p, varargin{:});
            
            subset = p.Results.subset;
            computeMeanIou = p.Results.computeMeanIou;
            cacheProbs = p.Results.cacheProbs;
            limitImageCount = p.Results.limitImageCount;
            
            epoch = numel(obj.stats.train);
            statsPath = fullfile(obj.nnOpts.expDir, sprintf('stats-%s-epoch%d.mat', subset, epoch));
            if exist(statsPath, 'file') && cacheProbs,
                % Get stats from disk
                statsStruct = load(statsPath, 'stats');
                stats = statsStruct.stats;
            else
                % Check that settings are valid
                if ~isinf(limitImageCount),
                    assert(~cacheProbs);
                end;
                
                % Limit images if specified (for quicker evaluation)
                if ~isinf(limitImageCount),
                    sel = randperm(numel(obj.imdb.data.test), min(limitImageCount, numel(obj.imdb.data.test)));
                    obj.imdb.data.test = obj.imdb.data.test(sel);
                end;
                
                % Get probabilities (softmaxed scores) for each region
                probsPath = fullfile(obj.nnOpts.expDir, sprintf('probs-%s-epoch%d.mat', subset, epoch));
                if exist(probsPath, 'file') && cacheProbs,
                    probsStruct = load(probsPath, 'probs');
                    probs = probsStruct.probs;
                else
                    % Init
                    imageCount = numel(obj.imdb.data.test); % even if we test on train it must say "test" here
                    probs = cell(imageCount, 1);
                    
                    % Set network to testing mode
                    outputVarIdx = obj.prepareNetForTest();
                    
                    for imageIdx = 1 : imageCount,
                        printProgress('Classifying images', imageIdx, imageCount, 10);
                        
                        % Check whether GT labels are available for this image
                        imageName = obj.imdb.data.test{imageIdx};
                        labelMap = obj.imdb.dataset.getImLabelMap(imageName);
                        if all(labelMap(:) == 0),
                            continue;
                        end;
                        
                        % Get batch
                        inputs = obj.imdb.getBatch(imageIdx, obj.net, obj.nnOpts);
                        
                        % Run forward pass
                        obj.net.eval(inputs);
                        
                        % Extract probs
                        curProbs = obj.net.vars(outputVarIdx).value;
                        curProbs = gather(reshape(curProbs, [size(curProbs, 3), size(curProbs, 4)]))';
                        
                        % Store
                        probs{imageIdx} = double(curProbs);
                    end;
                    
                    % Cache to disk
                    if cacheProbs,
                        fprintf('Saving probs to disk: %s\n', probsPath);
                        if varByteSize(probs) > 2e9,
                            matVersion = '-v7.3';
                        else
                            matVersion = '-v6';
                        end;
                        save(probsPath, 'probs', matVersion);
                    end;
                end;
                
                % Compute accuracy
                [pixAcc, meanClassPixAcc] = evaluatePixAcc(obj.imdb.dataset, obj.imdb.data.test, probs, obj.imdb.segmentFolderSP);
                
                % Compute meanIOU
                if computeMeanIou,
                    stats.meanIOU = evaluateMeanIOU(obj.imdb.data.test, probs, obj.imdb.segmentFolderSP);
                end;
                
                % Store results
                stats.pixAcc = pixAcc;
                stats.meanClassPixAcc = meanClassPixAcc;
                stats.trainLoss = obj.stats.train(end).objective;
                stats.valLoss   = obj.stats.val(end).objective;
                if cacheProbs,
                    if exist(statsPath, 'file'),
                        error('StatsPath already exists: %s', statsPath);
                    end;
                    save(statsPath, 'stats');
                end;
            end
        end
    end
    
    methods (Static)
        function stats = extractStats(net, ~)
            % stats = extractStats(net)
            %
            % Extract all losses from the network.
            % Contrary to CalvinNN.extractStats(..) this measures loss on
            % an image (subbatch) level, not on a region level!
            
            lossInds = find(cellfun(@(x) isa(x, 'dagnn.Loss'), {net.layers.block}));
            stats = struct();
            for lossIdx = 1 : numel(lossInds)
                layerIdx = lossInds(lossIdx);
                objective = net.layers(layerIdx).block.average ...
                    * net.layers(layerIdx).block.numAveraged ...
                    / net.layers(layerIdx).block.numSubBatches;
                assert(~isnan(objective));
                stats.(net.layers(layerIdx).outputs{1}) = objective;
            end
        end
        
        function[imageSamplingProbCum] = invFreqSumImageProbs(imdb, minImPixFreq, datasetMode)
            % [imageSamplingProbCum] = invFreqSumImageProbs(imdb, minImPixFreq, datasetMode)
            %
            % Compute a vector with the cumulative probs for each image.
            % This vector can directly be used to sample an image.
            % The weighting uses (possibly clamped) inverse frequency weights.
            %
            % Should be run for train and val, but not test.
            %
            % Copyright by Holger Caesar, 2015
            
            labelCount = imdb.numClasses;
            
            % Compute pixel-level label frequencies (also used without inv-freqs)
            labelImPixelFreqs = imdb.dataset.getLabelPixelFreqs();
            
            % Compute image-level frequencies
            % This is the simplified version of the frequencies used in the
            % RegionToPixel layer
            imageCount = numel(imdb.data.(datasetMode));
            imageSamplingProb = nan(imageCount, 1);
            for imageIdx = 1 : imageCount,
                printProgress('Computing image sampling prob. for image', imageIdx, imageCount);
                imageName = imdb.data.(datasetMode){imageIdx};
                labelMap = imdb.dataset.getImLabelMap(imageName);
                histo = histc(labelMap(:), 1:labelCount);
                nonZero = histo ~= 0;
                imageSamplingProb(imageIdx) = sum(histo(nonZero) ./ labelImPixelFreqs(nonZero));
            end;
            
            % Turn them into probabilities
            imageSamplingProb = imageSamplingProb / sum(imageSamplingProb);
            
            % Cutoff extreme sampling frequencies if required
            if ~isempty(minImPixFreq),
                imageSamplingProb = freqClampMinimum(imageSamplingProb, minImPixFreq);
            end;
            
            % Compute cumulate sum for lookup table
            imageSamplingProbCum = cumsum(imageSamplingProb);
        end
    end
end