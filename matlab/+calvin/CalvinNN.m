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
            obj.init(imdb, imdb.labelCount, nnOpts);
            
            % Load network and convert to DAG format
            obj.loadNetwork();
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
        state = accumulate_gradients(state, net, opts, batchSize, mmap);
        stats = accumulateStats(stats_);
        net = convertNetwork(net, imdb, nnOpts);
        stats = extractStats(net);
        epoch = findLastCheckpoint(modelDir);
        [net, stats] = loadState(fileName);
        mmap = map_gradients(fname, net, numGpus);
        stats = process_epoch(net, state, opts, mode);
        saveState(fileName, net, stats);
        write_gradients(mmap, net);
    end
end

