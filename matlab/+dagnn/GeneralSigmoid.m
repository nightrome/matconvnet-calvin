classdef GeneralSigmoid < dagnn.ElementWise
    % GeneralSigmoid
    % Implements a general sigmoid layer (i.e. sigmoid of a 1d linear function):
    % S(x) = 1 / (1 + exp(- (ax + b) ))
    %
    % For the derivatives check:
    % http://www.wolframalpha.com/input/?i=d%2Fdx+sigmoid%28ax%2Bb%29
    % http://www.wolframalpha.com/input/?i=d%2Fda+sigmoid%28ax%2Bb%29
    % http://www.wolframalpha.com/input/?i=d%2Fdb+sigmoid%28ax%2Bb%29
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        numClasses = 0;
    end
    
    methods
        function obj = GeneralSigmoid(varargin)
            obj.load(varargin) ;
        end
        
        function outputs = forward(obj, inputs, params)
            % Get inputs
            assert(numel(inputs) == 3);
            assert(numel(params) == 2);
            x = inputs{1};
            a = params{1};
            b = params{2};
            
            y = sigmoid(x, a, b);
            outputs{1} = y;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            x = inputs{1};
            a = params{1};
            b = params{2};
            
            y = sigmoid(x, a, b);
            dzdy = derOutputs{1};
            dzdb = dzdy .* (y .* (1 - y)); %   dzdy * dydb
            derInputs{1} = dzdb .* a; % dzdx = dzdy * dydx
            derParams{1} = dzdb .* x; % dzda = dzdy * dyda
            derParams{2} = dzdb;      % dzdb = dzdy * dydb
        end
        
        function params = initParams(obj)
            params{1} = repmat(single(-7), [obj.numClasses, 1]);
            params{2} = zeros([obj.numClasses, 1], 'single');
        end
    end
    
    methods (Access = private)
        function[y] = sigmoid(x, a, b)
            y = 1 ./ (1 + exp(- (a * x + b)));
        end
    end
end
