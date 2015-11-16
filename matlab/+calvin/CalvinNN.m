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
        opts
    end
    
    methods
        function obj = CalvinNN(imdb, opts)
            % obj = CalvinNN(imdb, [opts])
            
            % Set fields
            obj.imdb = imdb;
            
            % Init GPUs etc
            obj.init(imdb, imdb.getBatch, opts);
            
            % Load network and convert to DAG format
            obj.loadNetwork();
            
            % Perform training and validation
            obj.train();
        end
        
        function loadNetwork(obj)
            netIn = load(obj.imdb.netPath);
            
            if true, %TODO: determine whether it's a DAG
                obj.net = convertNetwork(netIn, obj.imdb, obj.opts);
            else
                obj.net = netIn;
            end;
        end
    end
    
    methods (Static)        
        obj = convertNetwork(net, imdb, opts);
    end
end

