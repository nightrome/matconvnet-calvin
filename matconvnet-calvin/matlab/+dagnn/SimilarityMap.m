classdef SimilarityMap < dagnn.Layer
    %
    % Executed on a pixel level.
    % For speedup this could be moved before the deconv layer.
    % Timings in the code are given for SimilarityMap right before loss
    % layer.
    %
    % inputs are: scoresClass
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
            assert(numel(inputs) == 1);
            scoresClass = inputs{1}; % don't move to CPU!
            
            %%% Compute softmax of scores
            X = scoresClass;
            ex = exp(X);
            z = sum(ex, 3);
            obj.scoresSoftMax = bsxfun(@rdivide, ex, z);
            
            %%% Replace probabilities by linear combination of probabilities
            if true
                % Copy from GPU to CPU
                scoresSoftMaxCPU = gather(obj.scoresSoftMax);
                
                % Quick mexed version (~47s)
                scoresWeighted = similarityMap_mapping(scoresSoftMaxCPU, single(obj.similarities));
                
                % Copy from CPU to GPU
                scoresWeighted = gpuArray(scoresWeighted);
            else
                % Slow Matlab version
                scoresWeighted = nan(size(obj.scoresSoftMax), 'like', obj.scoresSoftMax);
                for y = 1 : size(obj.scoresSoftMax, 1)
                    for x = 1 : size(obj.scoresSoftMax, 2)
                        if true
                            % Faster (140s)
                            scoresWeighted(y, x, :) = obj.similarities * squeeze(obj.scoresSoftMax(y, x, :));
                        elseif false
                            % Slow (~417min)
                            for zi = 1 : size(obj.scoresSoftMax, 3)
                                scoresWeighted(y, x, zi) = sum(squeeze(obj.scoresSoftMax(y, x, :)) .* obj.similarities(zi, :)');
                            end
                        elseif false
                            % Much slower *~328h)
                            for z1 = 1 : size(obj.scoresSoftMax, 3)
                                curSum = 0;
                                for z2 = 1 : size(obj.scoresSoftMax, 3)
                                    curSum = curSum + obj.scoresSoftMax(y, x, z2) * obj.similarities(z1, y2);
                                end
                                scoresWeighted(y, x, z1) = curSum;
                            end
                        end
                    end
                end
            end
            
            % Renormalize probabilities
            obj.renormFactors = sum(scoresWeighted, 3);
            obj.scoresWeightedNorm = bsxfun(@rdivide, scoresWeighted, obj.renormFactors);
            
            %%% Undo softmax
            scoresMixed = log(eps + obj.scoresWeightedNorm); % No need to multiply the weighting factor bsxfun(@times, obj.scoresWeightedNorm, z)
            assert(gather(~any(isinf(scoresMixed(:)))));
            
            % Create outputs
            outputs = cell(1, 1);
            outputs{1} = scoresMixed;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            dzdy = derOutputs{1};
            
            %%% Derive unsoftmax
            dzdx = dzdy ./ max(obj.scoresWeightedNorm, 1e-8);
            
            % Undo renormalization of probabilities
            dzdx = bsxfun(@rdivide, dzdx, obj.renormFactors);
            
            %%% Derive similarity mapping
            for y = 1 : size(dzdx, 1)
                for x = 1 : size(dzdx, 2)
                    dzdx(y, x, :) = obj.similarities * squeeze(dzdx(y, x, :));
                end
            end
            
            %%% Derive softmax
            dzdx = obj.scoresSoftMax .* bsxfun(@minus, dzdx, sum(dzdx .* obj.scoresSoftMax, 3));
            
            % Store gradients
            derInputs{1} = dzdx;
            derParams = {};
        end
        
        function obj = SimilarityMap(varargin)
            obj.load(varargin);
        end
    end
end