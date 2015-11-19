classdef RoiPooling < dagnn.Layer
    % Region of interest pooling layer.
    %
    % inputs are: convMaps, oriImSize, boxes
    % outputs are: rois, masks
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        poolSize = [1 1]
    end
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % Note: The mask is required here and (if existing) in the following
            % freeform layer.
            assert(numel(inputs) == 3);
            [outputs{1}, obj.mask] = roiPooling_wrapper(inputs{1}, inputs{2}, inputs{3}, obj.poolSize, true);
            outputs{2} = obj.mask;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            assert(numel(derOutputs) == 1);
            [~, ~, derInputs{1}] = roiPooling_wrapper(size(inputs{1}), inputs{2}, inputs{3}, obj.poolSize, false, obj.mask, derOutputs{1});
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function backwardAdvanced(obj, layer)
            %BACKWARDADVANCED Advanced driver for backward computation
            %  BACKWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
            %  the backward step of the layer.
            %
            %  The advanced interface can be changed in order to extend DagNN
            %  non-trivially, or to optimise certain blocks.
            %
            % Calvin: This layer needs to be modified as the output "label"
            % does not have a derivative and therefore backpropagation
            % would be skipped in the normal function.
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            
            % Modification:
            out = out(1);
            
            inputs = {net.vars(in).value};
            derOutputs = {net.vars(out).der};
            for i = 1:numel(derOutputs)
                if isempty(derOutputs{i}), return; end
            end
            
            if net.conserveMemory
                % clear output variables (value and derivative)
                % unless precious
                for i = out
                    if net.vars(i).precious, continue; end
                    net.vars(i).der = [];
                    net.vars(i).value = [];
                end
            end
            
            % compute derivatives of inputs and paramerters
            [derInputs, derParams] = obj.backward ...
                (inputs, {net.params(par).value}, derOutputs);
            
            % accumuate derivatives
            for i = 1:numel(in)
                v = in(i);
                if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
                    net.vars(v).der = derInputs{i};
                elseif ~isempty(derInputs{i})
                    net.vars(v).der = net.vars(v).der + derInputs{i};
                end
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1;
            end
            
            for i = 1:numel(par)
                p = par(i);
                if (net.numPendingParamRefs(p) == 0 && ~net.accumulateParamDers) ...
                        || isempty(net.params(p).der)
                    net.params(p).der = derParams{i};
                else
                    net.params(p).der = net.params(p).der + derParams{i};
                end
                net.numPendingParamRefs(p) = net.numPendingParamRefs(p) + 1;
            end
        end
        
        function obj = RoiPooling(varargin)
            obj.load(varargin);
        end
    end
end
