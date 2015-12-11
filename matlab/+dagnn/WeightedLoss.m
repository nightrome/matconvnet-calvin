classdef WeightedLoss < dagnn.Loss
    % WeightedLoss
    %
    % The same as dagnn.Loss, but allows to specify instanceWeights as an
    % additional input.
    %
    % inputs: scores, labels, instanceWeights
    % outputs: loss
    %
    % Note: If you use instanceWeights to change the total weight of this
    % batch, then you shouldn't use the default extractStatsFn anymore, a
    % it's average-loss depends on the number of boxes in the batch.
    %
    % Copyright by Holger Caesar, 2015
    
    properties (Transient)
        numBatches = 0;
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % Added a new input here called "lossInstanceWeights"
            
            % Get inputs
            assert(numel(inputs) == 3);
            scores = inputs{1};
            labels = inputs{2};
            instanceWeights = inputs{3};
            assert(~isempty(instanceWeights));
            assert(numel(instanceWeights) == size(scores, 4));
            assert(numel(instanceWeights) == size(labels, 4));
            
            % Compute loss
            outputs{1} = vl_nnloss(scores, labels, [], 'loss', obj.loss, 'instanceWeights', instanceWeights);
            
            % Update statistics
            n = obj.numAveraged;
            m = n + size(inputs{1}, 4); % only works when average sum(instanceWeights)~ 1
            obj.average = (n * obj.average + gather(outputs{1})) / m;
            obj.numAveraged = m;
            obj.numBatches = obj.numBatches + 1;
            assert(~isnan(obj.average));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            assert(numel(derOutputs) == 1);
            
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', inputs{3});
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function obj = WeightedLoss(varargin)
            obj = obj@dagnn.Loss(varargin{:});
        end
        
        function reset(obj)
            reset@dagnn.Loss(obj);
            obj.numBatches = 0;
        end
    end
end