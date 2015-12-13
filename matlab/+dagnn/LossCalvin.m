classdef LossCalvin < dagnn.Loss
    % Extension of Loss which *forces* the use of instanceWeights: If instanceWeights are
    % present in the network as a variable, it will use them.
    
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
            % Jasper: Indeed. Removed this option to enable not using instanceWeights
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
        
        function outputs = forward(obj, inputs, params)            
            % Get instanceWeights. For safety give error if unspecified
            inputNames = obj.net.layers(obj.layerIndex).inputs;
            [tf, iwInd] = ismember('instanceWeights', inputNames);
            if tf
                instanceWeights = inputs{iwInd};
            else
                error('Loss layer %d ignores instanceWeights.\n Set instanceWeight as input variable even if unused.', obj.layerIndex);                
            end
            
            outputs{1} = vl_nnloss(inputs{1}, inputs{2}, [], 'loss', obj.loss, 'instanceWeights', instanceWeights);
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Get instanceWeights
            inputNames = obj.net.layers(obj.layerIndex).inputs;
            [~, iwInd] = ismember('instanceWeights', inputNames);
            instanceWeights = inputs{iwInd};

            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', instanceWeights);
            derInputs{2} = [];
            derInputs{3} = []; 
            derParams = {};
        end
        
        function obj = LossCalvin(varargin)
            obj = obj@dagnn.Loss(varargin{:});
        end
    end
end
