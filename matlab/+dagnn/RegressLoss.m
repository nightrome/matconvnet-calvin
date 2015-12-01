classdef RegressLoss < dagnn.Loss
    
    methods
        function outputs = forward(obj, inputs, params)
            % Deal with NaNs in target scores which should be ignored
            isnanMask = isnan(inputs{2});
            regressionTargets = inputs{2};
            regressionTargets(isnanMask) = 0;
            regressionScore = inputs{1};
            regressionScore(isnanMask) = 0;
            
            outputs{1} = vl_nnloss_regress(regressionScore, regressionTargets, [], 'loss', obj.loss) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Deal with NaNs in target scores which should be ignored
            isnanMask = isnan(inputs{2});
            regressionTargets = inputs{2};
            regressionTargets(isnanMask) = 0;
            regressionScore = inputs{1};
            regressionScore(isnanMask) = 0;
            
            derInputs{1} = vl_nnloss_regress(regressionScore,regressionTargets, derOutputs{1}, 'loss', obj.loss) ;
            derInputs{2} = [] ;
            derInputs{3} = [] ;
            derParams = {} ;
        end
        
        function obj = RegressLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
