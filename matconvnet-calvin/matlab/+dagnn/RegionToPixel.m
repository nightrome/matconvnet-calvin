classdef RegionToPixel < dagnn.Layer
    % Go from a region level to a pixel level.
    % (to be able to compute a loss there)
    %
    % inputs are: scoresAll, regionToPixelAux
    % outputs are: scoresSP, labelsSP, weightsSP
    %
    % Note: 
    % - The presence/absence of regionToPixelAux.spLabelHistos indicates
    %   whether we are in train/val or test mode.
    % - For weakly supervised learning the outputs labelsSP and weightsSP
    %   are ignored.
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        inverseLabelFreqs = true;
        normalizeImageMass = true;
        replicateUnpureSPs = true;
        minPixFreq = []; % used only in getBatch
    end
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            assert(numel(inputs) == 2);
            [scoresSP, labelsSP, weightsSP, obj.mask] = regionToPixel_forward(inputs{1}, inputs{2}, obj.inverseLabelFreqs, obj.normalizeImageMass, obj.replicateUnpureSPs);
            
            % Split labels into labels and instance weights
            outputs = cell(3, 1);
            outputs{1} = scoresSP;
            outputs{2} = labelsSP;
            outputs{3} = weightsSP;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            % Go from a pixel level back to region level.
            % This uses the mask saved in the forward pass.
            %
            % Note: The gradients should already have an average
            % weighting of 'boxCount', as introduced in the forward
            % pass.
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            boxCount = size(inputs{1}, 4);
            dzdy = derOutputs{1};
            
            % Move inputs from GPU if necessary
            gpuMode = isa(dzdy, 'gpuArray');
            if gpuMode
                dzdy = gather(dzdy);
            end
            
            % Map SP gradients to RP+GT gradients
            dzdx = regionToPixel_backward(boxCount, obj.mask, dzdy);
            
            % Move outputs to GPU if necessary
            if gpuMode
                dzdx = gpuArray(dzdx);
            end
            
            % Store gradients
            derInputs{1} = dzdx;
            derInputs{2} = [];
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
        
        function obj = RegionToPixel(varargin)
            obj.load(varargin);
        end
    end
end