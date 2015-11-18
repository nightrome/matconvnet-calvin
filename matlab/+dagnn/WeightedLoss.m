classdef WeightedLoss < dagnn.Loss
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % Added a new input here called "lossInstanceWeights"
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, 'instanceWeights', inputs{3}) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', inputs{3}) ;
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {} ;
        end
        
        function obj = WeightedLoss(varargin)
            obj = obj@dagnn.Loss(varargin{:});
        end
    end
end
