classdef LabelPresence < dagnn.Layer
    % Convert pixel label scores to presence/absence scores for each class per batch.
    % (to be able to compute an image-level loss there)
    %
    % inputs are: scoresSP, labelImage
    % outputs are: scoresImage
    %
    % Copyright by Holger Caesar, 2015
    
    properties
    end
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            assert(numel(inputs) == 2);
            %             [scoresImage, labelsImage, weightsImage, obj.mask] = labelPresence_forward(inputs{1}, inputs{2});
            
            % Get inputs
            scoresSP = inputs{1};
            labelImage = inputs{2};
            
            % Move to CPU
            gpuMode = isa(scoresSP, 'gpuArray');
            if gpuMode
                scoresSP = gather(scoresSP);
            end
            
            labelList = unique(labelImage);
            labelListCount = numel(labelList);
            labelCount = size(scoresSP, 3);
            
            % Init
            scoresImage = nan(1, 1, labelCount, labelListCount); % score of the label, and all other labels
            obj.mask = nan(labelCount, labelListCount); % contains the label of each superpixel
            
            % Loss for each gt label
            [~, predictedLabels] = max(scoresSP, [], 3);
            for labelListIdx = 1 : labelListCount,
%                 label = labelList(labelListIdx);
                relRegions = 1:numel(predictedLabels);
                [scoresImage(1, 1, :, labelListIdx), relRegionsIdx] = max(scoresSP(:, :, :, relRegions), [], 4);
                spIdx = relRegions(relRegionsIdx);
                obj.mask(:, labelListIdx) = spIdx;
            end
            
%             %TODO: remove
%             scoresImage = scoresImage(:, :, :, 1);
%             obj.mask = obj.mask(:, 1);

%             % Loss for each predicted label
%             [~, predicted] = max(scoresSP, [], 3);
%             predictedUn = unique(predicted);
%             predictedCount = numel(predictedUn);
%             
%             % Init
%             scoresImage = nan(1, 1, labelCount, predictedCount); % score of the label, and all other labels
%             obj.mask = nan(labelCount, predictedCount); % contains the label of each superpixel
%             
%             for predictedIdx = 1 : numel(predicted)
%                 label = predictedUn(labelListIdx);
%                 relRegions = find(predicted == label);
%                 [scoresImage(1, 1, :, predictedIdx), relRegionsIdx] = max(scoresSP(:, :, label, relRegions), [], 4);
%                 spIdx = relRegions(relRegionsIdx);
%                 obj.mask(:, labelListIdx) = spIdx;
%             end
            
            % Convert outputs back to GPU if necessary
            if gpuMode
                scoresImage = gpuArray(scoresImage);
            end
            
            % Split labels into labels and instance weights
            outputs = cell(1, 1);
            outputs{1} = scoresImage;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            %
            % This uses the mask saved in the forward pass.
            
            assert(numel(derOutputs) == 1);
            
            % Get inputs
            spCount = size(inputs{1}, 4);
            dzdy = derOutputs{1};
            
            % Move inputs from GPU if necessary
            gpuMode = isa(dzdy, 'gpuArray');
            if gpuMode
                dzdy = gather(dzdy);
            end
            
            % Map Image gradients to RP+GT gradients
            dzdx = labelPresence_backward(spCount, obj.mask, dzdy);
            
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
        
        function obj = LabelPresence(varargin)
            obj.load(varargin);
        end
    end
end