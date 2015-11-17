classdef CalvinNN < handle
    % CalvinNN
    % Default network training script in Matconvnet-Calvin.
    % This uses Matconvnet's Directed Acyclic Graph structure and is
    % heavily inspired by the cnn_train_dag example.
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        net
        imdb
        nnOpts
        stats
    end
    
    methods
        function obj = CalvinNN(imdb, nnOpts)
            % obj = CalvinNN(imdb, [nnOpts])
            
            % Set fields
            obj.imdb = imdb;
            
            % Init GPUs etc
            obj.init(imdb, imdb.getBatch, nnOpts);
            
            % Load network and convert to DAG format
            obj.loadNetwork();
            
            % Perform training and validation
            obj.train();
        end
        
        function loadNetwork(obj)
            netIn = load(obj.imdb.netPath);
            
            % Convert net to DAG if necessary
            if isa(netIn, 'dagnn.DagNN'),
                obj.net = convertNetwork(netIn, obj.imdb, obj.nnOpts);
            else
                obj.net = netIn;
            end;
        end
    end
    
    methods (Static)        
        net = convertNetwork(net, imdb, nnOpts);
    end
end

