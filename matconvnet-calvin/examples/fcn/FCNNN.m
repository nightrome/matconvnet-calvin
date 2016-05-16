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
            
            fprintf('Converting test AlexNet-style network to train FCN (1x1 convolutions, loss, etc.)...\n');
            
            % Get initial model from VGG-VD-16
            obj.net = fcnInitializeModelGeneric(obj.imdb, 'sourceModelPath', obj.nnOpts.misc.netPath, ...
                'init', obj.nnOpts.misc.init, 'initLinCombPath', obj.nnOpts.misc.initLinCombPath, ...
                'enableCudnn', obj.nnOpts.misc.enableCudnn);
            if any(strcmp(obj.nnOpts.misc.modelType, {'fcn16s', 'fcn8s'}))
                % upgrade model to FCN16s
                obj.net = fcnInitializeModel16sGeneric(obj.imdb.numClasses, obj.net);
            end
            if strcmp(obj.nnOpts.misc.modelType, 'fcn8s')
                % upgrade model fto FCN8s
                obj.net = fcnInitializeModel8sGeneric(obj.imdb.numClasses, obj.net);
            end
            obj.net.meta.normalization.rgbMean = obj.imdb.imdb.rgbMean;
            obj.net.meta.classes = obj.imdb.imdb.classes.name;
            
            if obj.nnOpts.misc.weaklySupervised
                wsPresentWeight = 1 / (1 + wsUseAbsent);
                
                if obj.nnOpts.misc.wsEqualWeight
                    wsAbsentWeight = obj.imdb.numClasses * wsUseAbsent;
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
                accBlock = dagnn.SegmentationAccuracyFlexible('labelCount', obj.imdb.numClasses);
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
            addParameter(p, 'limitImageCount', Inf);
            addParameter(p, 'findMapping', false);
            parse(p, varargin{:});
            
            subset = p.Results.subset;
            limitImageCount = p.Results.limitImageCount;
            findMapping = p.Results.findMapping;
            
            % Set the datasetMode to be active
            if strcmp(subset, 'test'),
                temp = [];
            else
                temp = obj.imdb.data.test;
                obj.imdb.data.test = obj.imdb.data.(subset);
            end
            
            % Run test
            stats = obj.test('subset', subset, 'limitImageCount', limitImageCount, 'findMapping', findMapping);
            
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
            
            % Disable accuracy layer
            accuracyIdx = obj.net.getLayerIndex('accuracy');
            if ~isnan(accuracyIdx),
                obj.net.removeLayer('accuracy');
            end;
            
            % Disable softmax layer (that is before labelpresence)
            softmaxIdx = obj.net.getLayerIndex('softmax');
            if ~isnan(softmaxIdx),
                obj.net.removeLayer('softmax');
            end;
            
            % Remove loss or replace by normal softmax
            lossIdx = find(cellfun(@(x) isa(x, 'dagnn.Loss'), {obj.net.layers.block}));
            lossName = obj.net.layers(lossIdx).name;
            lossType = obj.net.layers(lossIdx).block.loss;
            lossInputs = obj.net.layers(lossIdx).inputs;
            if strcmp(lossType, 'softmaxlog'),
                obj.net.removeLayer(lossName);
                outputLayerName = 'softmax';
                outputVarName = 'scores';
                obj.net.addLayer(outputLayerName, dagnn.SoftMax(), lossInputs{1}, outputVarName, {});
                outputVarIdx = obj.net.getVarIndex(outputVarName);
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
            addParameter(p, 'limitImageCount', Inf);
            addParameter(p, 'doCache', true);
            addParameter(p, 'findMapping', false);
            addParameter(p, 'plotFreq', 15);
            addParameter(p, 'printFreq', 30);
            addParameter(p, 'showPlot', false);
            addParameter(p, 'doOutputMaps', true);
            parse(p, varargin{:});
            
            subset = p.Results.subset;
            limitImageCount = p.Results.limitImageCount;
            doCache = p.Results.doCache;
            findMapping = p.Results.findMapping;
            plotFreq = p.Results.plotFreq;
            printFreq = p.Results.printFreq;
            showPlot = p.Results.showPlot;
            doOutputMaps = p.Results.doOutputMaps;
            
            epoch = numel(obj.stats.train);
            statsPath = fullfile(obj.nnOpts.expDir, sprintf('stats-%s-epoch%d.mat', subset, epoch));
            labelingDir = fullfile(obj.nnOpts.expDir, sprintf('labelings-%s-epoch-%d', subset, epoch));
            mapOutputFolder = fullfile(obj.nnOpts.expDir, sprintf('outputMaps-epoch-%d', epoch));
            if exist(statsPath, 'file'),
                % Get stats from disk
                statsStruct = load(statsPath, 'stats');
                stats = statsStruct.stats;
            else
                % Limit images if specified (for quicker evaluation)
                if ~isinf(limitImageCount),
                    sel = randperm(numel(obj.imdb.data.test), min(limitImageCount, numel(obj.imdb.data.test)));
                    obj.imdb.data.test = obj.imdb.data.test(sel);
                end;
                
                % Set network to testing mode
                outputVarIdx = obj.prepareNetForTest();
                
                % Create output folder
                if doOutputMaps && ~exist(mapOutputFolder, 'dir')
                    mkdir(mapOutputFolder)
                end
                if ~exist(labelingDir, 'dir')
                    mkdir(labelingDir);
                end
                
                % Prepare stuff for visualization
                labelNames = obj.imdb.dataset.getLabelNames();
                colorMapping = FCNNN.labelColors(obj.imdb.numClasses);
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
                    colorMappingPred = FCNNN.labelColors(numel(labelNamesPred));
                    %                     assert(obj.imdb.numClasses == obj.imdb.dataset.labelCount); %obj.imdb.labelCount should correspond to the target dataset
                else
                    % Normal test mode
                    labelNamesPred = labelNames;
                    colorMappingPred = colorMapping;
                end
                
                % Init
                evalTimer = tic;
                imageCount = numel(obj.imdb.data.test); % even if we test on train it must say "test" here
                confusion = zeros(obj.imdb.numClasses, numel(labelNamesPred));
                
                for imageIdx = 1 : imageCount,                    
                    % Get batch
                    inputs = obj.imdb.getBatch(imageIdx, obj.net, obj.nnOpts);
                    
                    % Get labelMap
                    imageName = obj.imdb.data.(obj.imdb.datasetMode){imageIdx};
                    labelMap = uint16(obj.imdb.batchOpts.imageNameToLabelMap(imageName));
                    if obj.imdb.batchOpts.translateLabels,
                        % Before: 255 = ignore, 0 = bkg, 1:n = classes
                        % After : 0 = ignore, 1 = bkg, 2:n+1 = classes
                        labelMap = mod(labelMap + 1, 256);
                    end;
                    % 0 = ignore, 1:n = classes
                    
                    % Run forward pass
                    obj.net.eval(inputs);
                    
                    % Forward image through net and get predictions
                    scores = obj.net.vars(outputVarIdx).value;
                    [~, outputMap] = max(scores, [], 3);
                    outputMap = gather(outputMap);
                    outputMap = imresize(outputMap, size(labelMap), 'method', 'nearest');
                    
                    % Accumulate errors
                    ok = labelMap > 0;
                    confusion = confusion + accumarray([labelMap(ok), outputMap(ok)], 1, size(confusion));
                    
                    % If a folder was specified, output the predicted label maps
                    if doOutputMaps
                        outputPath = fullfile(mapOutputFolder, [imageName, '.mat']);
                        if obj.imdb.numClasses > 200
                            save(outputPath, 'outputMap');
                        else
                            save(outputPath, 'outputMap', 'scores');
                        end
                    end;
                    
                    % Plot example images
                    if mod(imageIdx - 1, plotFreq) == 0 || imageIdx == imageCount
                        
                        % Print segmentation
                        if showPlot,
                            figure(100);
                            clf;
                            FCNNN.displayImage(obj.imdb.numClasses, rgb / 255, labelMap, outputMap);
                            drawnow;
                        end;
                        
                        % Create tiled image with image+gt+outputMap
                        if true
                            if obj.imdb.dataset.annotation.labelOneIsBg
                                skipLabelInds = 1;
                            else
                                skipLabelInds = [];
                            end;
                            
                            % Create tiling
                            tile = ImageTile();
                            
                            % Add GT image
                            image = obj.imdb.dataset.getImage(imageName) * 255;
                            tile.addImage(image / 255);
                            labelMapIm = ind2rgb(double(labelMap), colorMapping);
                            labelMapIm = imageInsertBlobLabels(labelMapIm, labelMap, labelNames, 'skipLabelInds', skipLabelInds);
                            tile.addImage(labelMapIm);
                            
                            % Add prediction image
                            outputMapNoBg = outputMap;
                            outputMapNoBg(labelMap == 0) = 0;
                            outputMapIm = ind2rgb(outputMapNoBg, colorMappingPred);
                            outputMapIm = imageInsertBlobLabels(outputMapIm, outputMapNoBg, labelNamesPred, 'skipLabelInds', skipLabelInds);
                            tile.addImage(outputMapIm);
                            
                            % Highlight differences between GT and outputMap
                            if ~findMapping
                                errorMap = ones(size(labelMap));
                                if obj.imdb.dataset.annotation.labelOneIsBg
                                    % Datasets where bg is 1 and void is 0 (i.e. VOC)
                                    tooMuch = labelMap ~= outputMap & labelMap == 1 & outputMap >= 2;
                                    tooFew  = labelMap ~= outputMap & labelMap >= 2 & outputMap == 1;
                                    rightClass = labelMap == outputMap & labelMap >= 2 & outputMap >= 2;
                                    wrongClass = labelMap ~= outputMap & labelMap >= 2 & outputMap >= 2;
                                    errorMap(tooMuch) = 2;
                                    errorMap(tooFew) = 3;
                                    errorMap(rightClass) = 4;
                                    errorMap(wrongClass) = 5;
                                else
                                    % For datasets without bg
                                    rightClass = labelMap == outputMap & labelMap >= 1;
                                    wrongClass = labelMap ~= outputMap & labelMap >= 1;
                                    errorMap(rightClass) = 4;
                                    errorMap(wrongClass) = 5;
                                end
                                errorIm = ind2rgb(double(errorMap), colorMappingError);
                                tile.addImage(errorIm);
                            end
                            
                            % Save segmentation
                            image = tile.getTiling('totalX', numel(tile.images), 'delimiterPixels', 1, 'backgroundBlack', false);
                            imPath = fullfile(labelingDir, [imageName, '.png']);
                            imwrite(image, imPath);
                        end
                    end
                    
                    % Print message
                    if mod(imageIdx - 1, printFreq) == 0 || imageIdx == imageCount
                        evalTime = toc(evalTimer);
                        fprintf('Processing image %d of %d (%.2f Hz)...\n', imageIdx, imageCount, imageIdx / evalTime);
                    end
                end;
                
                if findMapping
                    % Save mapping to disk
                    mappingPath = fullfile(obj.nnOpts.expDir, sprintf('mapping-%s.mat', subset));
                    save(mappingPath, 'confusion');
                else
                    % Final statistics, remove classes missing in test
                    % Note: Printing statistics earlier does not make sense if we remove missing
                    % classes
                    [stats.iu, stats.miu, stats.pacc, stats.macc] = FCNNN.getAccuracies(confusion);
                    stats.confusion = confusion;
                    fprintf('Result with all classes:\n');
                    fprintf('IU %4.1f ', 100 * stats.iu);
                    fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
                        100 * stats.miu, 100 * stats.pacc, 100 * stats.macc);
                    
                    % Save results
                    if doCache
                        save(statsPath, '-struct', 'stats');
                    end
                end
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
        
        function [IU, meanIU, pixelAccuracy, meanAccuracy] = getAccuracies(confusion)
            pos = sum(confusion, 2);
            res = sum(confusion, 1)';
            tp = diag(confusion);
            IU = tp ./ max(1, pos + res - tp);
            missing = pos == 0;
            meanIU = mean(IU(~missing));
            pixelAccuracy = sum(tp) / max(1, sum(confusion(:)));
            meanAccuracy = mean(tp(~missing) ./ pos(~missing));
        end
        
        function displayImage(colorCount, im, lb, outputMap)
            subplot(2, 2, 1);
            image(im);
            axis image;
            title('source image');
            
            subplot(2, 2, 2);
            image(uint8(lb - 1));
            axis image;
            title('ground truth')
            
            cmap = FCNNN.labelColors(colorCount);
            subplot(2, 2, 3);
            image(uint8(outputMap - 1));
            axis image;
            title('predicted');
            
            colormap(cmap);
        end
        
        function cmap = labelColors(colorCount)
            cmap = zeros(colorCount, 3);
            for i = 1 : colorCount
                id = i-1;
                r = 0;
                g = 0;
                b = 0;
                for j=0:7
                    r = bitor(r, bitshift(bitget(id, 1), 7 - j));
                    g = bitor(g, bitshift(bitget(id, 2), 7 - j));
                    b = bitor(b, bitshift(bitget(id, 3), 7 - j));
                    id = bitshift(id, -3);
                end
                cmap(i, 1) = r;
                cmap(i, 2) = g;
                cmap(i,3) = b;
            end
            cmap = cmap / 255;
        end
    end
end