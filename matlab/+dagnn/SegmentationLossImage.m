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
        instanceWeights
        isPresent
        softmax
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            
            %%%% Get inputs
            assert(numel(inputs) == 3);
            scoresMap = inputs{1};
            labelsImageCell = inputs{2};
            classWeights = inputs{3};
            labelCount = size(scoresMap, 3);
            imageCount = size(scoresMap, 4);
            sampleCount = labelCount * imageCount;
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
                obj.mask = nan(sampleCount, 1); % contains the coordinates of the pixel with highest score per class
                
                % Process each image/crop separately
                for imageIdx = 1 : imageCount
                    % For each label, get the scores of the highest scoring pixel
                    labelMax = max(max(scoresMap(:, :, :, imageIdx), [], 1), [], 2);
                    
                    for labelIdx = 1 : labelCount
                        offset = (imageIdx-1) * labelCount;
                        sampleIdx = offset + labelIdx;
                        
                        [y, x] = find(scoresMap(:, :, labelIdx, imageIdx) == labelMax(labelIdx), 1); % always take first pix with max score
                        obj.scoresImage(:, :, :, sampleIdx) = scoresMap(y, x, :, imageIdx);
                        obj.mask(sampleIdx, 1) = y + (x - 1) * size(scoresMap, 1);
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
                obj.isPresent = ismember(1:sampleCount, presentInds)';
                
                % Compute loss
                % To simplify things we remove the subtraction of Xmax from
                % X, which is supposed to reduce numerical problems
                ex = exp(bsxfun(@minus, X, max(X, [], 3)));
                sumEx = sum(ex, 3);
                obj.softmax = bsxfun(@rdivide, ex, sumEx);
                obj.softmax(:, :, :, ~obj.isPresent) = 1 - obj.softmax(:, :, :, ~obj.isPresent);
                obj.softmax = max(obj.softmax, 1e-10);
                t = -log(obj.softmax(ci));
                obj.instanceWeights = ones(1, 1, 1, sampleCount);
                
                % Weight per class
                if ~isempty(classWeights)
                    obj.instanceWeights = obj.instanceWeights .* classWeights(labelsDummy);
                end
                
                % Renormalize present labels per image
                sampleImage = repmatEach(1:imageCount, labelCount);
                presentWeight = 1 / (1 + obj.useAbsent); % give all or half of the weight to presence
                for imageIdx = 1 : imageCount
                    sel = sampleImage == imageIdx & obj.isPresent;
                    obj.instanceWeights(sel) = obj.instanceWeights(sel) ./ (sum(obj.instanceWeights(sel)) / presentWeight);
                end
                
                % Renormalize or disable absent labels per image
                for imageIdx = 1 : imageCount,
                    sel = sampleImage == imageIdx & ~obj.isPresent;
                    
                    if obj.useAbsent
                        obj.instanceWeights(sel) = obj.instanceWeights(sel) ./ (sum(obj.instanceWeights(sel)) / presentWeight);
                    else
                        obj.instanceWeights(sel) = 0;
                    end
                end
                
                loss = sum(t .* obj.instanceWeights);
            end;
            
            %%%% Assign outputs
            outputs{1} = loss;
            
            % Update statistics
            assert(~isnan(loss) && ~isinf(loss));
            n = obj.numAveraged;
            m = n + imageCount;
            obj.average = (n * obj.average + double(gather(loss))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            %%%% Get inputs
            assert(numel(inputs) == 3);
            scoresMap = inputs{1};
            imageSizeY = size(scoresMap, 1);
            labelCount = size(scoresMap, 3);
            imageCount = size(scoresMap, 4);
            labelsDummy = repmat(1:labelCount, [1, imageCount]);
            labelsDummy = reshape(labelsDummy, 1, 1, 1, []);
            
            assert(numel(derOutputs) == 1);
            dzdOutput = derOutputs{1};
            
            %%%% Loss derivatives
            X = obj.softmax;
            c = labelsDummy;
            
            inputSize = [size(X, 1), size(X, 2), size(X, 3), size(X, 4)];
            labelSize = [size(c, 1), size(c, 2), size(c, 3), size(c, 4)];
            assert(isequal(labelSize(1:2), inputSize(1:2)));
            assert(labelSize(4) == inputSize(4));
            
            % from category labels to indexes
            numPixelsPerImage = prod(inputSize(1:2));
            numPixels = numPixelsPerImage * inputSize(4);
            imageVolume = numPixelsPerImage * inputSize(3);
            
            n = reshape(0:numPixels-1, labelSize);
            offset = 1 + mod(n, numPixelsPerImage) + ...
                imageVolume * fix(n / numPixelsPerImage);
            ci = offset + numPixelsPerImage * max(c - 1,0);
            
            % Weight gradients per instance
            dzdOutput = dzdOutput * obj.instanceWeights;
            
            % Compute gradients for log-loss
            dzdLoss = zeros(size(X), 'like', X);
            dzdLoss(ci) = - dzdOutput ./ max(X(ci), 1e-8);
            
            % Compute gradients for softmax
            X = obj.scoresImage;
            ex = exp(bsxfun(@minus, X, max(X, [], 3)));
            sumEx = sum(ex, 3);
            softmaxDef = bsxfun(@rdivide, ex, sumEx);
            dzdImage = softmaxDef .* bsxfun(@minus, dzdLoss, sum(dzdLoss .* softmaxDef, 3));
            
            %%%% Map gradients from image to pixel-level
            dzdxMap = zeros(size(scoresMap), 'single');
            for imageIdx = 1 : imageCount
                for labelIdx = 1 : labelCount
                    offset = (imageIdx-1) * labelCount;
                    sampleIdx = offset + labelIdx;
                    pos = obj.mask(sampleIdx, 1);
                    x = 1 + floor((pos-1) / imageSizeY);
                    y = pos - (x-1) * imageSizeY;
                    
                    dzdxMap(y, x, :, imageIdx) = dzdxMap(y, x, :, imageIdx) + dzdImage(1, 1, :, sampleIdx);
                end
            end
            
            % Move outputs to GPU if necessary
            gpuMode = isa(inputs{1}, 'gpuArray');
            if gpuMode
                dzdxMap = gpuArray(dzdxMap);
            end
            
            %%%% Assign outputs
            derInputs{1} = dzdxMap;
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
