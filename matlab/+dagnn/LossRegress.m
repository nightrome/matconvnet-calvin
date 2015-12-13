classdef LossRegress < dagnn.Loss
    properties
        smoothMaxDiff = 1; % For smooth-loss (see vl_nnloss_regress)
    end
    
    methods
        function forwardAdvanced(obj, layer)
            %FORWARDADVANCED  Advanced driver for forward computation
            %  FORWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
            %  the forward step of the layer.
            %
            %  The advanced interface can be changed in order to extend DagNN
            %  non-trivially, or to optimise certain blocks.
            %
            % Jasper: Overrides standard forward pass to avoid giving up when any of
            % the inputs is empty.
            
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            par = layer.paramIndexes ;
            net = obj.net ;
            
            inputs = {net.vars(in).value} ;
            
            % give up if any of the inputs is empty (this allows to run
            % subnetworks by specifying only some of the variables as input --
            % however it is somewhat dangerous as inputs could be legitimaly
            % empty)
%              if any(cellfun(@isempty, inputs)), return ; end
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1 ;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious & ~net.computingDerivative & net.conserveMemory
                        net.vars(v).value = [] ;
                    end
                end
            end
            
            %[net.vars(out).value] = deal([]) ;
            
            % call the simplified interface
            outputs = obj.forward(inputs, {net.params(par).value}) ;
            [net.vars(out).value] = deal(outputs{:}) ;
        end
        
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
        
        function obj = LossRegress(varargin)
            obj.load(varargin) ;
        end
    end
end
