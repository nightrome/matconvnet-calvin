classdef FCNNN < CalvinNN
    % FCNNN Fully Convolutional Network class implemented as subclass of
    % CalvinNN.
    %
    % Copyright by Holger Caesar, 2016
    
    methods
        function obj = FCNNN(net, imdb, nnOpts)
            obj = obj@CalvinNN(net, imdb, nnOpts);
        end
        
        function convertNetwork(obj)
            % convertNetwork(obj)
            %
            % Does not call the equivalent function in CalvinNN.
            
            % Get initial model from VGG-VD-16
            obj.net = fcnInitializeModelGeneric(obj.imdb.imdb, 'sourceModelPath', obj.nnOpts.misc.netPath, ...
                'init', obj.nnOpts.misc.init, 'initLinCombPath', obj.nnOpts.misc.initLinCombPath, ...
                'enableCudnn', obj.nnOpts.misc.enableCudnn);
            if any(strcmp(obj.nnOpts.misc.modelType, {'fcn16s', 'fcn8s'}))
                % upgrade model to FCN16s
                obj.net = fcnInitializeModel16sGeneric(obj.imdb.imdb.labelCount, obj.net);
            end
            if strcmp(obj.nnOpts.misc.modelType, 'fcn8s')
                % upgrade model fto FCN8s
                obj.net = fcnInitializeModel8sGeneric(obj.imdb.imdb.labelCount, obj.net);
            end
            obj.net.meta.normalization.rgbMean = obj.imdb.imdb.rgbMean;
            obj.net.meta.classes = obj.imdb.imdb.classes.name;
            
            if obj.nnOpts.misc.weaklySupervised
                wsPresentWeight = 1 / (1 + wsUseAbsent);
                
                if obj.nnOpts.misc.wsEqualWeight
                    wsAbsentWeight = obj.imdb.imdb.labelCount * wsUseAbsent;
                else
                    wsAbsentWeight = 1 - wsPresentWeight;
                end
            else
                wsPresentWeight = [];
                wsAbsentWeight = [];
            end
            
            % Replace unweighted loss layer
            layerFS = dagnn.SegmentationLossPixel();
            layerWS = dagnn.SegmentationLossImage('useAbsent', obj.nnOpts.misc.wsUseAbsent, 'useScoreDiffs', obj.nnOpts.misc.wsUseScoreDiffs, 'presentWeight', wsPresentWeight, 'absentWeight', wsAbsentWeight);
            objIdx = obj.net.getLayerIndex('objective');
            assert(strcmp(obj.net.layers(objIdx).block.loss, 'softmaxlog'));
            
            % Add a layer that automatically decides whether to use FS or WS
            layerSS = dagnn.SegmentationLossSemiSupervised('layerFS', layerFS, 'layerWS', layerWS);
            layerSSInputs = [obj.net.layers(objIdx).inputs, {'labelsImage', 'classWeights', 'isWeaklySupervised', 'masksThingsCell'}];
            layerSSOutputs = obj.net.layers(objIdx).outputs;
            obj.net.removeLayer('objective');
            obj.net.addLayer('objective', layerSS, layerSSInputs, layerSSOutputs, {});
            
            % Accuracy layer
            if obj.imdb.dataset.annotation.hasPixelLabels
                % Replace accuracy layer with 21 classes by flexible accuracy layer
                accIdx = obj.net.getLayerIndex('accuracy');
                accLayer = obj.net.layers(accIdx);
                accInputs = accLayer.inputs;
                accOutputs = accLayer.outputs;
                accBlock = dagnn.SegmentationAccuracyFlexible('labelCount', obj.imdb.imdb.labelCount);
                obj.net.removeLayer('accuracy');
                obj.net.addLayer('accuracy', accBlock, accInputs, accOutputs, {});
            else
                % Remove accuracy layer if no pixel-level labels exist
                obj.net.removeLayer('accuracy');
            end
            
            % Sort layers by their first occurrence
            sortLayers(obj.net);
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
            
            %             % Move to GPU
            %             if ~isempty(obj.nnOpts.gpus),
            %                 obj.net.move('gpu');
            %             end;
            %
            %             % Enable test mode
            %             obj.imdb.setDatasetMode('test');
            %             obj.net.mode = 'test';
            %
            %             % Reset segments to default
            %             obj.imdb.batchOpts.segments.colorTypeIdx = 1;
            %             obj.imdb.updateSegmentNames();
            %
            %             % Get pixel output variable name
            %             regiontopixelIdx = obj.net.getLayerIndex('regiontopixel8');
            %             regiontopixelOutput = obj.net.layers(regiontopixelIdx).outputs{1};
            %
            %             % Disable labelpresence layer (these needs to happen before we
            %             % remove the softmax layer)
            %             labelpresenceIdx = obj.net.getLayerIndex('labelpresence');
            %             if ~isnan(labelpresenceIdx),
            %                 obj.net.removeLayer('labelpresence');
            %                 softmaxlossIdx = obj.net.getLayerIndex('softmaxloss');
            %                 obj.net.layers(softmaxlossIdx).inputs{1} = regiontopixelOutput;
            %                 obj.net.rebuild();
            %             end;
            %             % Disable softmax layer (that is before labelpresence)
            %             softmaxIdx = obj.net.getLayerIndex('softmax');
            %             if ~isnan(softmaxIdx),
            %                 obj.net.removeLayer('softmax');
            %             end;
            %
            %             % Replace softmaxloss by softmax
            %             lossIdx = find(cellfun(@(x) isa(x, 'dagnn.Loss'), {obj.net.layers.block}));
            %             lossName = obj.net.layers(lossIdx).name;
            %             lossType = obj.net.layers(lossIdx).block.loss;
            %             lossInputs = obj.net.layers(lossIdx).inputs;
            %             if strcmp(lossType, 'softmaxlog'),
            %                 obj.net.removeLayer(lossName);
            %                 outputLayerName = 'softmax';
            %                 obj.net.addLayer(outputLayerName, dagnn.SoftMax(), lossInputs{1}, 'scores', {});
            %                 outputLayerIdx = obj.net.getLayerIndex(outputLayerName);
            %                 outputVarIdx = obj.net.layers(outputLayerIdx).outputIndexes;
            %             elseif strcmp(lossType, 'log'),
            %                 % Only output the scores of the regiontopixel layer
            %                 obj.net.removeLayer(lossName);
            %                 outputVarIdx = obj.net.getVarIndex(obj.net.getOutputs{1});
            %             else
            %                 error('Error: Unknown loss function!');
            %             end;
            %             assert(numel(outputVarIdx) == 1);
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
    end
end