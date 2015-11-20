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
        function obj = CalvinNN(net, imdb, nnOpts)
            % obj = CalvinNN(imdb, [nnOpts])
            
            % Set fields
            obj.imdb = imdb;
            
            % Init options and GPUs
            obj.init(nnOpts);
            
            % Load network and convert to DAG format
            obj.loadNetwork(net);
        end
        
        function loadNetwork(obj, netIn)
            % Load the network from file if necessary
            if ischar(netIn),
                netIn = load(netIn);
            end
            
            % Store the network
            if isfield(netIn, 'net')
                if isa(netIn.net, 'dagnn.DagNN')
                    % The network is already a DAG
                    obj.net = netIn.net;
                elseif isfield(netIn.net, 'vars')
                    % Convert a stored DAG to a proper class object
                    obj.net = dagnn.DagNN.loadobj(netIn.net);
                else
                    error('Error: Network is neither in SimpleNN nor DAG format!');
                end
            elseif isfield(netIn, 'layers'),
                % Convert a simpleNN network to DAG format
                obj.convertNetwork(netIn);
            else
                error('Error: Network is neither in SimpleNN nor DAG format!');
            end
        end
        
        % Declarations for methods that are in separate files
        convertNetwork(obj, net);
        convertNetworkToFastRcnn(obj);
        init(obj, varargin);
        saveState(obj, fileName);
        train(obj);
        test(obj, targetEpoch);
    end
    
    methods (Access = protected)
        % Declarations for methods that are in separate files
        stats = accumulateStats(obj, stats_);
        state = accumulate_gradients(obj, state, net, batchSize, mmap);
        loadState(obj, fileName);
        mmap = map_gradients(fname, net, numGpus);
        stats = process_epoch(obj, net, state);
        write_gradients(mmap, net);
    end
    
    methods (Static)
        stats = extractStats(net);
        epoch = findLastCheckpoint(modelDir);
    end
end