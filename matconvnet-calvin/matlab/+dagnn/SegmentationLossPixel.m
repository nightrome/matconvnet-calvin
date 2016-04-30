classdef SegmentationLossPixel < dagnn.Loss
    % SegmentationLossPixel
    %
    % Similar to dagnn.SegmentationLoss, but also sets pixel weights.
    %
    % Inputs: scoresMap, labels, classWeights
    % Outputs: loss
    %
    % Note: All weights can be empty, which means they are ignored.
    % Note: If you use this for weakly supervised, the loss output will be
    % wrong (divided by instances, not images)
    %
    % Copyright by Holger Caesar, 2016
    
    properties (Transient)
        instanceWeights
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>

            % Get inputs
            assert(numel(inputs) == 3);
            scores = inputs{1};
            labels = inputs{2};
            classWeights = inputs{3};
            
            % Compute invMass
            mass = sum(sum(labels > 0, 2), 1); % Removed the +1
            invMass = zeros(size(mass));
            nonEmpty = mass ~= 0;
            invMass(nonEmpty) = 1 ./ mass(nonEmpty);
            
            % Compute pixelWeights
            if isempty(classWeights)
                pixelWeights = [];
            else
                classWeightsPad = [0; classWeights(:)];
                
                %%% Pixel weighting
                pixelWeights = classWeightsPad(labels + 1);
                
                % Make sure mass of the image does not change
                curMasses = sum(sum(pixelWeights, 1), 2);
                divisor = curMasses ./ mass;
                nonZero = mass ~= 0;
                pixelWeights(:, :, :, nonZero) = bsxfun(@rdivide, pixelWeights(:, :, :, nonZero), divisor(nonZero));
                assert(all(abs(sum(sum(pixelWeights, 1), 2) - mass) < 1e-6));
            end;
            
            % Combine mass invMass and pixelWeights in instanceWeights
            obj.instanceWeights = invMass;
            if ~isempty(pixelWeights)
                obj.instanceWeights = bsxfun(@times, obj.instanceWeights, pixelWeights);
            end
                
            % Checks
            if ~isempty(obj.instanceWeights)
                assert(~any(isnan(obj.instanceWeights(:))))
            end
            
            outputs{1} = vl_nnloss(scores, labels, [], ...
                'loss', obj.loss, ...
                'instanceWeights', obj.instanceWeights);
            
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
            
            derInputs{1} = vl_nnloss(scores, labels, derOutputs{1}, ...
                'loss', obj.loss, ...
                'instanceWeights', obj.instanceWeights);
            derInputs{2} = [];
            derInputs{3} = [];
            derInputs{4} = [];
            derParams = {};
        end
        
        function obj = SegmentationLossPixel(varargin)
            obj.load(varargin);
        end
        
        function forwardAdvanced(obj, layer)
            % Modification: Overrides standard forward pass to avoid giving up when any of
            % the inputs is empty.
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            inputs = {net.vars(in).value};
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious && ~net.computingDerivative && net.conserveMemory
                        net.vars(v).value = [];
                    end
                end
            end
            
            % call the simplified interface
            outputs = obj.forward(inputs, {net.params(par).value});
            for oi = 1:numel(out)
                net.vars(out(oi)).value = outputs{oi};
            end
        end
    end
end