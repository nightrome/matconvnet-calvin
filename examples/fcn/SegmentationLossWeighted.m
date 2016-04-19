classdef SegmentationLossWeighted < dagnn.Loss
    % Same as SegmentationLoss, but allows external instanceWeights as
    % inputs{3}. Doesn't do anything so far
    %
    % Inputs: scores, labels, instanceWeights
    % Outputs: loss
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            assert(numel(inputs) == 2 || numel(inputs) == 3);
            
            % Compute instanceWeights
            mass = sum(sum(inputs{2} > 0, 2), 1) + 1;
            instanceWeights = 1 ./ mass;
            if numel(inputs) == 3
                pixelWeights = inputs{3};
                assert(~any(isnan(pixelWeights(:))))
            else
                pixelWeights = [];
            end
            
            outputs{1} = vl_nnloss_pixelweighted(inputs{1}, inputs{2}, [], ...
                'loss', obj.loss, ...
                'instanceWeights', instanceWeights, ...
                'pixelWeights', pixelWeights);
            assert(gather(~isnan(outputs{1})));
            n = obj.numAveraged;
            m = n + size(inputs{1}, 4);
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            % Compute instanceWeights
            mass = sum(sum(inputs{2} > 0,2),1) + 1;
            instanceWeights = 1./mass;
            if numel(inputs) == 3
                pixelWeights = inputs{3};
            else
                pixelWeights = [];
            end
            
            
            derInputs{1} = vl_nnloss_pixelweighted(inputs{1}, inputs{2}, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', instanceWeights, ...
                'pixelWeights', pixelWeights);
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function obj = SegmentationLoss(varargin)
            obj.load(varargin);
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
            % Jasper: Indeed. Removed this option to enable not using pixelWeights
            %              if any(cellfun(@isempty, inputs)), return; end
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious && ~net.computingDerivative && net.conserveMemory
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
