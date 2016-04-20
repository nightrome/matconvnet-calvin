classdef SegmentationLossWeighted < dagnn.Loss
    % SegmentationLossWeighted
    %
    % Same as SegmentationLoss, but allows additional weights.
    %
    % Inputs: scores, labels, pixelWeights, instanceWeights
    % Outputs: loss
    %
    % Note: All weights can be empty, which means they are ignored.
    %
    % Copyright by Holger Caesar, 2016
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            assert(numel(inputs) == 4);
            
            % Get inputs
            scores = inputs{1};
            labels = inputs{2};
            pixelWeights = inputs{3};
            instanceWeights = inputs{4};
            
            % Compute instanceWeights
            mass = sum(sum(labels > 0, 2), 1); % Removed the +1
            invMass = zeros(size(mass));
            nonEmpty = mass ~= 0;
            invMass(nonEmpty) = 1 ./ mass(nonEmpty);
            if isempty(instanceWeights)
                instanceWeights = invMass;
            else
                instanceWeights = instanceWeights .* invMass;
            end
                
            % Checks
            if ~isempty(pixelWeights)
                assert(~any(isnan(pixelWeights(:))))
            end
            
            outputs{1} = vl_nnloss_pixelweighted(scores, labels, [], ...
                'loss', obj.loss, ...
                'instanceWeights', instanceWeights, ...
                'pixelWeights', pixelWeights);
            assert(gather(~isnan(outputs{1})));
            n = obj.numAveraged;
            m = n + size(scores, 4);
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m;
            obj.numAveraged = m;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            % Get inputs
            scores = inputs{1};
            labels = inputs{2};
            pixelWeights = inputs{3};
            instanceWeights = inputs{4};
            
            % Compute instanceWeights
            mass = sum(sum(labels > 0, 2), 1) + 1;
            invMass = 1 ./ mass;
            if isempty(instanceWeights)
                instanceWeights = invMass;
            else
                instanceWeights = instanceWeights .* invMass;
            end
            
            derInputs{1} = vl_nnloss_pixelweighted(scores, labels, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', instanceWeights, ...
                'pixelWeights', pixelWeights);
            derInputs{2} = [];
            derInputs{3} = [];
            derInputs{4} = [];
            derParams = {};
        end
        
        function obj = SegmentationLoss(varargin) %#ok<STOUT>
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
