classdef RegressLoss < dagnn.Loss
    properties
        smoothMaxDiff = 1; % For smooth-loss (see vl_nnloss_regress)
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % Deal with NaNs in target scores which should be ignored
            regressionTargets = inputs{2};
            isnanMask = isnan(regressionTargets);
            regressionTargets(isnanMask) = 0;
            regressionScore = squeeze(inputs{1});
            assert(isequal(size(regressionTargets), size(regressionScore)));
            regressionScore(isnanMask) = 0;
            
            % Get instanceWeights if specified
            inputNames = obj.net.layers(obj.layerIndex).inputs;
            [tf, iwInd] = ismember('instanceWeights', inputNames);
            if tf
                instanceWeights = inputs{iwInd};
            else
                instanceWeights = [];
            end
            
            % Get loss
            outputs{1} = vl_nnloss_regress(regressionScore, regressionTargets, [], ... 
                'loss', obj.loss, 'smoothMaxDiff', obj.smoothMaxDiff, 'instanceWeights', instanceWeights);
            
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            % Deal with NaNs in target scores which should be ignored
            regressionTargets = inputs{2};
            isnanMask = isnan(regressionTargets);
            regressionTargets(isnanMask) = 0;
            regressionScore = squeeze(inputs{1});
            assert(isequal(size(regressionTargets), size(regressionScore)));
            regressionScore(isnanMask) = 0;
            
            % Get instanceWeights if specified
            inputNames = obj.net.layers(obj.layerIndex).inputs;
            [tf, iwInd] = ismember('instanceWeights', inputNames);
            if tf
                instanceWeights = inputs{iwInd};
            else
                instanceWeights = [];
            end
            
            % Get gradient
            derInputs{1} = vl_nnloss_regress(regressionScore,regressionTargets, derOutputs{1}, ...
                'loss', obj.loss, 'smoothMaxDiff', obj.smoothMaxDiff, 'instanceWeights', instanceWeights);

            derInputs{1} = reshape(derInputs{1}, size(inputs{1}));
            derInputs{2} = [] ;
            derInputs{3} = [] ;
            derParams = {} ;
        end
        
        function obj = RegressLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
