classdef ROIPooling < dagnn.Filter
  % Region of interest pooling layer. 
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
      % inputs are: convMaps, oriImSize, boxes
      % outputs are: rois, masks
      % Note: The mask is required here and (if existing) in the following
      % freeform layer.
      assert(numel(inputs) == 3);
      [outputs{1}, obj.mask] = roiPooling_wrapper(inputs{1}, inputs{2}, inputs{3}, obj.poolSize, true);
      outputs{2} = obj.mask;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
      % inputs are: convMaps, oriImSize, boxes
      assert(numel(inputs) == 3);
      [~, ~, derInputs{1}] = roiPooling_wrapper(size(inputs{1}), inputs{2}, inputs{3}, obj.poolSize, false, obj.mask, derOutputs{1});
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
        %TODO: Check whether this is correct
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = inputSizes{1}(3) ;
    end

    function obj = ROIPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
