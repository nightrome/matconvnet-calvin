classdef RegressLoss < dagnn.Loss
    properties
        smoothMaxDiff = 1; % For smooth-loss (see vl_nnloss_regress)
    end
    
    methods
        function outputs = forward(obj, inputs, params)
            % Deal with NaNs in target scores which should be ignored
            regressionTargets = inputs{2};
            isnanMask = isnan(regressionTargets);
            regressionTargets(isnanMask) = 0;
            regressionScore = permute(inputs{1}, [4 3 2 1]);
            regressionScore(isnanMask) = 0;
            
            outputs{1} = vl_nnloss_regress(regressionScore, regressionTargets, [], 'loss', obj.loss, 'smoothMaxDiff', obj.smoothMaxDiff) ;
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Deal with NaNs in target scores which should be ignored
            regressionTargets = inputs{2};
            isnanMask = isnan(regressionTargets);
            regressionTargets(isnanMask) = 0;
            regressionScore = permute(inputs{1}, [4 3 2 1]);
            regressionScore(isnanMask) = 0;
            
            derInputs{1} = vl_nnloss_regress(regressionScore,regressionTargets, derOutputs{1}, 'loss', obj.loss, 'smoothMaxDiff', obj.smoothMaxDiff) ;
            derInputs{2} = [] ;
            derInputs{3} = [] ;
            derParams = {} ;
        end
        
        function obj = RegressLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
