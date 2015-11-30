classdef WeightedLoss < dagnn.Loss
    % WeightedLoss
    %
    % The same as dagnn.Loss, but allows to specify instanceWeights as an
    % additional input.
    %
    % inputs: scores, labels, instanceWeights
    % outputs: weighted loss
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
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, 'instanceWeights', inputs{3});
            n = obj.numAveraged;
            m = n + size(inputs{1},4);
            obj.average = (n * obj.average + gather(outputs{1})) / m;
            obj.numAveraged = m;
            
            obj.numBatches = obj.numBatches + 1;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
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
            numBatches = 0;
        end
    end
end