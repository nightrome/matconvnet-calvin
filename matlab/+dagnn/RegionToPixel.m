classdef RegionToPixel < dagnn.Filter
    % Go from a region level to a pixel level.
    % (to be able to compute a loss there)
    %
    % Copyright by Holger Caesar, 2015
    
    properties
    end
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % inputs are: scoresAll, batchAux
            % outputs are: scoresSP, labelsSP
            assert(numel(inputs) == 2);
            [outputs{1}, labels, obj.mask] = e2s2_train_regiontopixel_forward(inputs{1}, inputs{2});
            
            % TODO: remove this and introduce sample-level weights in the
            % loss
            labels(:, :, 2, :) = [];
            outputs{2} = labels;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            % Go from a pixel level back to region level.
            % This uses the mask saved in the forward pass.
            %
            % Note: The gradients should already have an average
            % weighting of 'boxCount', as introduced in the forward
            % pass.
            
            % inputs are: boxCount
            assert(numel(inputs) == 1);
            
            % Get inputs
            boxCount = inputs{1};
            dzdx = derOutputs{1};
            
            % Move inputs from GPU if necessary
            gpuMode = isa(dzdx, 'gpuArray');
            if gpuMode,
                dzdx = gather(dzdx);
            end;
            
            % Map SP gradients to RP+GT gradients
            dzdxout = e2s2_train_regiontopixel_backward(boxCount, obj.mask, dzdx);
            
            % Move outputs to GPU if necessary
            if gpuMode,
                dzdxout = gpuArray(dzdxout);
            end;
            
            % Store gradients
            derInputs{1} = dzdxout;
            derParams = {};
        end
        
        function kernelSize = getKernelSize(obj) %#ok<MANU>
            kernelSize = [1, 1];
        end
        
        function outputSizes = getOutputSizes(obj, inputSizes)
            %TODO: Check whether this is correct
            outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes);
            outputSizes{1}(3) = inputSizes{1}(3);
        end
        
        function obj = RegionToPixel(varargin)
            obj.load(varargin);
        end
    end
end
