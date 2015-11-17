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
            
            % Init options and GPUs
            obj.init(nnOpts);
            
            % Load network and convert to DAG format
            obj.loadNetwork();
        end
        
        function loadNetwork(obj)
            netIn = load(obj.nnOpts.netPath);
            
            % Convert net to DAG if necessary
            if ~isa(netIn, 'dagnn.DagNN'),
                obj.convertNetwork(netIn);
            else
                obj.net = netIn;
            end;
        end
        
        % Declaration for methods that are in separate files
        convertNetwork(obj, net);
        init(obj, varargin);
        train(obj);
    end
    
    methods (Static)
        state = accumulate_gradients(state, net, opts, batchSize, mmap);
        stats = accumulateStats(stats_);
        stats = extractStats(net);
        epoch = findLastCheckpoint(modelDir);
        [net, stats] = loadState(fileName);
        mmap = map_gradients(fname, net, numGpus);
        stats = process_epoch(net, state, imdb, opts, mode);
        saveState(fileName, net, stats);
        write_gradients(mmap, net);
    end
end