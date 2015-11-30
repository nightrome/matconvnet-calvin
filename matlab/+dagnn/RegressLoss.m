classdef RegressLoss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      lossWeight = repmat(double(inputs{3} ~= 1), [1 4]);
      inputs{2} = inputs{2} .* lossWeight;
      inputs{1} = inputs{1} .* permute(lossWeight, [4 3 2 1]);
        
      outputs{1} = vl_nnloss_regress(inputs{1}, inputs{2}, [], 'loss', obj.loss) ;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      lossWeight = repmat(double(inputs{3} ~= 1), [1 4]);
      inputs{2} = inputs{2} .* lossWeight;
      inputs{1} = inputs{1} .* permute(lossWeight, [4 3 2 1]);

      derInputs{1} = vl_nnloss_regress(inputs{1}, inputs{2}, derOutputs{1}, 'loss', obj.loss) ;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
    end
    
    function obj = RegressLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
