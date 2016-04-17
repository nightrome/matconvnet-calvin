classdef SegmentationLabelPresence < dagnn.Layer
    % Convert pixel-level class scores to presence/absence scores for each class per image/crop.
    % (to be able to compute an image-level loss there)
    %
    % inputs are: scoresMap, labelImages
    % outputs are: scoresImages
    %
    % Copyright by Holger Caesar, 2015
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            
            % Get inputs
            assert(numel(inputs) == 2);
            scoresMap = inputs{1};
            labelImages = inputs{2};
            
            % Move to CPU
            gpuMode = isa(scoresMap, 'gpuArray');
            if gpuMode
                scoresMap = gather(scoresMap);
            end
            assert(~any(isnan(scoresMap(:))))
            
            % Count total number of samples and init scores
            labelCount = size(scoresMap, 3);
            sampleCount = numel(cell2mat(labelImages));
            scoresImages = nan(1, 1, labelCount, sampleCount, 'single');
            obj.mask = nan(labelCount, sampleCount); % contains the label scores of each GT label
            
            % Process each image/crop separately
            imageCount = numel(labelImages);
            totalSampleOffset = 0;
            for imageIdx = 1 : imageCount
                
                % Init
                labelList = unique(labelImages{imageIdx});
                curSampleCount = numel(labelList);
                
                % For each label, get the scores of the highest scoring pixel
                for curSampleIdx = 1 : curSampleCount,
                    labelIdx = labelList(curSampleIdx);
                    curScoresMap = scoresMap(:, :, labelIdx, imageIdx);
                    [y, x] = find(curScoresMap == max(curScoresMap(:)), 1); % always take first pix with max score
                    scoresImages(:, :, :, curSampleIdx+totalSampleOffset) = scoresMap(y, x, :, imageIdx);
                    obj.mask(:, curSampleIdx+totalSampleOffset) = y + (x - 1) * size(scoresMap, 1);
                end
                totalSampleOffset = totalSampleOffset + curSampleCount;
            end
            
            % Convert outputs back to GPU if necessary
            if gpuMode
                scoresImages = gpuArray(scoresImages);
            end
            
            % Store outputs
            outputs = cell(1, 1);
            outputs{1} = scoresImages;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            %
            % This uses the mask saved in the forward pass.
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            scoresMap = inputs{1};
            labelImages = inputs{2};
            imageSizeY = size(scoresMap, 1);
            imageSizeX = size(scoresMap, 2);
            labelCount = size(scoresMap, 3);
            imageCount = numel(labelImages);
            dzdy = derOutputs{1};
            
            % Move inputs from GPU if necessary
            gpuMode = isa(dzdy, 'gpuArray');
            if gpuMode
                dzdy = gather(dzdy);
            end
            
            % Map Image gradients to RP+GT gradients
            dzdxSamples = segmentationLabelPresence_backward(imageSizeY, imageSizeX, obj.mask, dzdy);
            
            % Sum the per-sammple gradients of each image
            dzdx = nan(imageSizeY, imageSizeX, labelCount, imageCount, 'single');
            totalSampleOffset = 0;
            for imageIdx = 1 : imageCount,
                samples = totalSampleOffset + 1 : totalSampleOffset + numel(labelImages{imageIdx});
                dzdx(:, :, :, imageIdx) = nansum(dzdxSamples(:, :, :, samples), 4);
                totalSampleOffset = totalSampleOffset + numel(labelImages{imageIdx});
            end;
            
            % Move outputs to GPU if necessary
            if gpuMode
                dzdx = gpuArray(dzdx);
            end
            
            % Store gradients
            derInputs{1} = dzdx;
            derInputs{2} = [];
            derParams = {};
        end
        
        function backwardAdvanced(obj, layer)
            %BACKWARDADVANCED Advanced driver for backward computation
            %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
            %  the backward step of the layer.
            %
            %  The advanced interface can be changed in order to extend DagNN
            %  non-trivially, or to optimise certain blocks.
            %
            % Calvin: This layer needs to be modified as the output "label"
            % does not have a derivative and therefore backpropagation
            % would be skipped in the normal function.
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            
            % Modification:
            out = out(1);
            
            inputs = {net.vars(in).value};
            derOutputs = {net.vars(out).der};
            for i = 1:numel(derOutputs)
                if isempty(derOutputs{i}), return; end
            end
            
            if net.conserveMemory
                % clear output variables (value and derivative)
                % unless precious
                for i = out
                    if net.vars(i).precious, continue; end
                    net.vars(i).der = [];
                    net.vars(i).value = [];
                end
            end
            
            % compute derivatives of inputs and paramerters
            [derInputs, derParams] = obj.backward ...
                (inputs, {net.params(par).value}, derOutputs);
            
            % accumuate derivatives
            for i = 1:numel(in)
                v = in(i);
                if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
                    net.vars(v).der = derInputs{i};
                elseif ~isempty(derInputs{i})
                    net.vars(v).der = net.vars(v).der + derInputs{i};
                end
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1;
            end
            
            for i = 1:numel(par)
                p = par(i);
                if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
                        || isempty(net.params(p).der)
                    net.params(p).der = derParams{i};
                else
                    net.params(p).der = net.params(p).der + derParams{i};
                end
                net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1;
            end
        end
        
        function obj = SegmentationLabelPresence(varargin)
            obj.load(varargin);
        end
    end
end