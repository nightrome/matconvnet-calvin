classdef LossWeighted < dagnn.Loss
    % LossWeighted
    %
    % The same as dagnn.Loss, but (optionally) allows to specify
    % instanceWeights as an additional input.
    %
    % inputs: scores, labels, [instanceWeights]
    % outputs: loss
    %
    % Note: If you use instanceWeights to change the total weight of this
    % batch, then you shouldn't use the default extractStatsFn anymore, as
    % its average-loss depends on the number of boxes in the batch.
    %
    % Copyright by Holger Caesar, 2015
    
    properties (Transient)
        numSubBatches = 0;
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % Added a new input here called "lossInstanceWeights"
            
            % Get inputs
            assert(numel(inputs) == 3);
            scores = inputs{1};
            labels = inputs{2};
            instanceWeights = inputs{3};
            if ~isempty(instanceWeights)
                assert(numel(instanceWeights) == size(scores, 4));
                assert(numel(instanceWeights) == size(labels, ndims(labels)));
            end
            
            % Compute loss
            outputs{1} = vl_nnloss(scores, labels, [], 'loss', obj.loss, 'instanceWeights', instanceWeights);
            
            % Update statistics
            n = obj.numAveraged;
            m = n + size(inputs{1}, 4);
            obj.average = (n * obj.average + gather(outputs{1})) / m;
            obj.numAveraged = m;
            obj.numSubBatches = obj.numSubBatches + 1;
            assert(~isnan(obj.average));
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            assert(numel(derOutputs) == 1);
            
            derInputs{1} = vl_nnloss(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss, 'instanceWeights', inputs{3});
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function obj = LossWeighted(varargin)
            obj = obj@dagnn.Loss(varargin{:});
        end
        
        function reset(obj)
            reset@dagnn.Loss(obj);
            obj.numSubBatches = 0;
        end
        
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
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            
            inputs = {net.vars(in).value};
            
            % give up if any of the inputs is empty (this allows to run
            % subnetworks by specifying only some of the variables as input --
            % however it is somewhat dangerous as inputs could be legitimaly
            % empty)
            % Jasper: Indeed. Removed this option to enable not using instanceWeights
            %              if any(cellfun(@isempty, inputs)), return; end
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious & ~net.computingDerivative & net.conserveMemory
                        net.vars(v).value = [];
                    end
                end
            end
            
            %[net.vars(out).value] = deal([]);
            
            % call the simplified interface
            outputs = obj.forward(inputs, {net.params(par).value});
            [net.vars(out).value] = deal(outputs{:});
        end
    end
end