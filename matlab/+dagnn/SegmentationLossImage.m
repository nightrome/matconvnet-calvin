classdef SegmentationLossImage < dagnn.Loss
    % SegmentationLossImage
    %
    % Same as SegmentationLossWeighted, but weakly supervised following
    % "What's the point: Semantic segmentation with point supervision" by
    % Russakovsky et al., arxiv 2015
    %
    % Inputs: scoresMap, labelsImage, classWeights
    % Outputs: loss
    %
    % Note: All weights can be empty, which means they are ignored.
    %
    % Copyright by Holger Caesar, 2016

    
    properties
        useAbsent = true;
    end
    
    properties (Transient)
        mask
        scoresImage
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            
            %%%% Get inputs
            assert(numel(inputs) == 3);
            scoresMap = inputs{1};
            labelsImageCell = inputs{2};
            classWeights = inputs{3};
            labelCount = size(scoresMap, 3);
            sampleCount = labelCount * imageCount;
            imageCount = numel(labelsImageCell);
            labelsDummy = repmat(1:labelCount, [1, imageCount]);
            labelsDummy = reshape(labelsDummy, 1, 1, 1, []);

            %%%% Pixel to image mapping
            % Move to CPU
            gpuMode = isa(scoresMap, 'gpuArray');
            if gpuMode
                scoresMap = gather(scoresMap);
            end
            assert(~any(isnan(scoresMap(:))))
            
            if true
                % Count total number of samples and init scores
                obj.scoresImage = nan(1, 1, labelCount, sampleCount, 'single');
                obj.mask = nan(labelCount, labelCount); % contains the coordinates of the pixel with highest score per class
                
                % Process each image/crop separately
                for imageIdx = 1 : imageCount
                    % For each label, get the scores of the highest scoring pixel
                    labelMax = max(max(scoresMap(:, :, :, imageIdx), [], 1), [], 2);
                    
                    for labelIdx = 1 : labelCount
                        [y, x] = find(scoresMap(:, :, labelIdx, imageIdx) == labelMax(labelIdx), 1); % always take first pix with max score
                        offset = (imageIdx-1) * labelCount;
                        obj.scoresImage(:, :, :, offset+labelIdx) = scoresMap(y, x, :, imageIdx);
                        obj.mask(:, labelIdx) = y + (x - 1) * size(scoresMap, 1);
                    end
                end
            end
            
            %%% Loss function from vl_nnloss
            if true,
                X = obj.scoresImage;
                c = labelsDummy;
                
                % from category labels to indexes
                inputSize = [size(X,1) size(X,2) size(X,3) size(X,4)];
                labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)];
                numPixelsPerImage = prod(inputSize(1:2));
                numPixels = numPixelsPerImage * inputSize(4);
                imageVolume = numPixelsPerImage * inputSize(3);
                
                n = reshape(0:numPixels-1, labelSize);
                offset = 1 + mod(n, numPixelsPerImage) + ...
                    imageVolume * fix(n / numPixelsPerImage);
                ci = offset + numPixelsPerImage * max(c - 1, 0);
                
                % Get presence/absence info
                presentInds = cell(imageCount, 1);
                for imageIdx = 1 : imageCount,
                    presentInds{imageIdx} = labelsImageCell{imageIdx} + (imageIdx-1) * labelCount;
                end;
                presentInds = cell2mat(presentInds);
                isPresent = ismember(1:sampleCount, presentInds)';
                
                % Compute loss
                % To simplify things we remove the subtraction of Xmax from
                % X, which is supposed to reduce numerical problems
                ex = exp(X);
                sumExp = sum(ex, 3);
                softmax = bsxfun(@rdivide, ex, sumExp);
                softmax(:, :, :, ~isPresent) = 1 - softmax(:, :, :, ~isPresent);
                t = -log(softmax(ci));
                instanceWeights = ones(1, 1, 1, sampleCount);
                
                % Weight per class
                if ~isempty(classWeights)
                    instanceWeights = instanceWeights .* classWeights(labelsDummy);
                end
                
                % Renormalize present labels per image
                sampleImage = repmatEach(1:imageCount, labelCount);
                presentWeight = 1 / (1 + obj.useAbsent); % give all or half of the weight to presence
                for imageIdx = 1 : imageCount
                    sel = sampleImage == imageIdx & isPresent;
                    instanceWeights(sel) = instanceWeights(sel) ./ (sum(instanceWeights(sel)) / presentWeight);
                end
                
                % Renormalize or disable absent labels per image
                for imageIdx = 1 : imageCount,
                    sel = sampleImage == imageIdx & ~isPresent;
                    
                    if obj.useAbsent
                        instanceWeights(sel) = instanceWeights(sel) ./ (sum(instanceWeights(sel)) / presentWeight);
                    else
                        instanceWeights(sel) = 0;
                    end
                end
                
                loss = sum(t .* instanceWeights);
                
%                 if true
%                     presentInds = cell(imageCount, 1);
%                     for imageIdx = 1 : imageCount,
%                         presentInds{imageIdx} = labelsImageCell{imageIdx} + (imageIdx-1) * labelCount;
%                     end;
%                     presentInds = cell2mat(presentInds);
%                     presentT = log(sum(ex(:, :, :, presentInds), 3)) - X(ci(:, :, :, presentInds));
%                     
%                     presentWeights = ones(size(presentT)) / numel(presentT);
%                     if ~isempty(classWeights)
%                         presentWeights = presentWeights .* classWeights(presentInds);
%                     end
%                     presentLoss = sum(presentT .* presentWeights);
%                     loss = presentLoss;
%                 end
%                 
%                 if obj.useAbsent
%                     absentInds = cell(imageCount, 1);
%                     for imageIdx = 1 : imageCount,
%                         absentInds{imageIdx} = labelsImageCell{imageIdx} + (imageIdx-1) * labelCount;
%                     end;
%                     absentInds = cell2mat(absentInds);
%                     absentSumExp = sumExp(:, :, :, absentInds);
%                     absentT = log(absentSumExp) - log(absentSumExp - ex(ci(:, :, :, absentInds)));
%                     
%                     absentWeights = ones(size(absentT)) / numel(absentT);
%                     if ~isempty(classWeights)
%                         absentWeights = absentWeights .* classWeights(absentInds);
%                     end
%                     absentLoss = sum(absentT .* absentWeights);
%                     loss = (loss + absentLoss) / 2;
%                 end
            end;
            
            %%%% Assign outputs
            outputs{1} = loss;
            
            % Update statistics
            assert(gather(~isnan(loss)));
            n = obj.numAveraged;
            m = n + imageCount;
            obj.average = (n * obj.average + double(gather(loss))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            %%%% Get inputs
            assert(numel(inputs) == 3);
            scoresMap = inputs{1};
            labelsImageCell = inputs{2};
            classWeights = inputs{3};
            labelCount = size(scoresMap, 3);
            imageCount = numel(labelsImageCell);
            labelsDummy = repmat(1:labelCount, [1, imageCount]);
            labelsDummy = reshape(labelsDummy, 1, 1, 1, []);
            
            assert(numel(derOutputs) == 1);
            dzdLoss = derOutputs{1};
            
            %%%% Loss derivatives
            
            X = obj.scoresImage;
            c = labelsDummy;
            
            inputSize = [size(X,1) size(X,2) size(X,3) size(X,4)];
            labelSize = [size(c,1) size(c,2) size(c,3) size(c,4)];
            assert(isequal(labelSize(1:2), inputSize(1:2)));
            assert(labelSize(4) == inputSize(4));
            
            % from category labels to indexes
            numPixelsPerImage = prod(inputSize(1:2));
            numPixels = numPixelsPerImage * inputSize(4);
            imageVolume = numPixelsPerImage * inputSize(3);
            
            n = reshape(0:numPixels-1,labelSize);
            offset = 1 + mod(n, numPixelsPerImage) + ...
                imageVolume * fix(n / numPixelsPerImage);
            ci = offset + numPixelsPerImage * max(c - 1,0);
            
            % Compute gradients
%             dzdLoss = dzdLoss * instanceWeights;
                        
            Xmax = max(X,[],3);
            ex = exp(bsxfun(@minus, X, Xmax));
            dzdImage = bsxfun(@rdivide, ex, sum(ex,3));
            dzdImage(ci) = dzdImage(ci) - 1;
            dzdImage = bsxfun(@times, dzdLoss, dzdImage);
            
            %%%% Map gradients from image to pixel-level
            imageSizeY = size(scoresMap, 1);
            imageSizeX = size(scoresMap, 2);
            labelCount = size(scoresMap, 3);
            imageCount = numel(labelsImageCell);
            
            % Map Image gradients to RP+GT gradients
            dzdxSamples = segmentationLabelPresence_backward(imageSizeY, imageSizeX, obj.mask, dzdImage);
            
            % Sum the per-sammple gradients of each image
            dzdx = nan(imageSizeY, imageSizeX, labelCount, imageCount, 'single');
            totalSampleOffset = 0;
            for imageIdx = 1 : imageCount,
                samples = totalSampleOffset + 1 : totalSampleOffset + numel(labelsImageCell{imageIdx});
                dzdx(:, :, :, imageIdx) = nansum(dzdxSamples(:, :, :, samples), 4);
                totalSampleOffset = totalSampleOffset + numel(labelsImageCell{imageIdx});
            end;
            
            % Move outputs to GPU if necessary
            gpuMode = isa(inputs{1}, 'gpuArray');
            if gpuMode
                dzdx = gpuArray(dzdx);
            end
            
            %%%% Assign outputs
            derInputs{1} = dzdx;
            derInputs{2} = [];
            derInputs{3} = [];
            derInputs{4} = [];
            derParams = {};
        end
        
        function obj = SegmentationLoss(varargin) %#ok<STOUT>
            obj.load(varargin);
        end
        
        function forwardAdvanced(obj, layer)
            % Modification: Overrides standard forward pass to avoid giving up when any of
            % the inputs is empty.
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            
            inputs = {net.vars(in).value};
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious && ~net.computingDerivative && net.conserveMemory
                        net.vars(v).value = [];
                    end
                end
            end
            
            % call the simplified interface
            outputs = obj.forward(inputs, {net.params(par).value});
            [net.vars(out).value] = deal(outputs{:});
        end
    end
end
