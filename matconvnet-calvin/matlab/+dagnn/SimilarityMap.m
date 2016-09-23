classdef SimilarityMap < dagnn.Layer
    %
    % Executed on a pixel level.
    % Has to be after the deconv layer (slow!) unless we have a downsized
    % version of the labels.
    %
    % inputs are: scoresClass, labels
    % outputs are: scoresMixed
    %
    % Copyright by Holger Caesar, 2016
    
    properties
        similarities
    end
    
    properties (Transient)
        scoresSoftMax
        scoresWeightedNorm
        renormFactors
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            
            % Get inputs
            assert(numel(inputs) == 2);
            scoresClass = inputs{1};
            labels = inputs{2};
            
            % TODO: move to CPU?
            
            %%% Compute softmax of scores
            X = scoresClass;
            ex = exp(X);
            z = sum(ex, 3); % TODO: we can probably remove z everywhere
            obj.scoresSoftMax = bsxfun(@rdivide, ex, z);
            
            %%% Replace exponents by linearcombination of exponent            
            % Semi-vectorized (offset time and RAM usage) ~0.5s
            [m, n] = size(obj.similarities);
            similaritiesR = reshape(obj.similarities, 1, m, n);
            scoresWeighted = nan(size(obj.scoresSoftMax), 'like', obj.scoresSoftMax);
            for y = 1 : size(obj.scoresSoftMax, 1)
                gts = max(1, labels(y, :)); % just assign the first class if the pixel is irrelevant
                scoresWeighted(y, :, :) = obj.scoresSoftMax(y, :, :) .* similaritiesR(1, gts, :);
            end
            
            % Renormalize probabilities
            obj.renormFactors = sum(scoresWeighted, 3);
            obj.scoresWeightedNorm = bsxfun(@rdivide, scoresWeighted, obj.renormFactors);
            
            %%% Undo softmax
            scoresMixed = log(bsxfun(@mtimes, obj.scoresWeightedNorm, z));
            
            % Create outputs
            outputs = cell(1, 1);
            outputs{1} = scoresMixed;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            dzdy = derOutputs{1};
            labels = inputs{2};
            
            %%% Derive unsoftmax
            dzdx = dzdy ./ max(obj.scoresWeightedNorm, 1e-8);
            
            % Undo renormalization
            dzdx = bsxfun(@rdivide, dzdx, obj.renormFactors);
            
            %%% Derive similarity mapping
            [m, n] = size(obj.similarities);
            similaritiesR = reshape(obj.similarities, 1, m, n);
            for y = 1 : size(dzdx, 1)
                gts = max(1, labels(y, :)); % just assign the first class if the pixel is irrelevant
                dzdx(y, :, :) = dzdx(y, :, :) .* similaritiesR(1, gts, :);
            end
            
            %%% Derive softmax
            dzdx = obj.scoresSoftMax .* bsxfun(@minus, dzdx, sum(dzdx .* obj.scoresSoftMax, 3));
            
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
            
            % Modification:n2
            
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
        
        function obj = SimilarityMap(varargin)
            obj.load(varargin);
        end
    end
end