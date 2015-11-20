function results = test(obj, targetEpoch)
% Test function
%
% - Loads the network within this function using targetEpoch or checking HD for last epoch saved
% - Does a single processing of an epoch for testing
% - Uses the nnOpts.extractStatsTestFn function for the testing
% - Automatically changes softmaxloss to softmax. Other losses are not yet supported

tempExtractStatsFn = obj.nnOpts.extractStatsFn;
obj.nnOpts.extractStatsFn = obj.nnOpts.extractStatsTestFn;

% setup GPUs
numGpus = numel(obj.nnOpts.gpus);
if numGpus > 1,
    pool = gcp('nocreate');
    
    % Delete parpool with wrong size
    if ~isempty(pool) && pool.NumWorkers ~= numGpus,
        delete(gcp);
    end;
    
    % Create new parpool
    if isempty(pool) || ~pool.isvalid(),
        parpool('local',numGpus);
    end
    
    % Assign GPUs
    spmd, gpuDevice(obj.nnOpts.gpus(labindex)), end    
    
elseif numGpus == 1,
    gpuDevice(obj.nnOpts.gpus);
end;

% Load correct network (Latest if targetEpoch is not given)
modelPath = @(ep) fullfile(obj.nnOpts.expDir, sprintf('net-epoch-%d.mat', ep));
if nargin == 1
    targetEpoch = CalvinNN.findLastCheckpoint(obj.nnOpts.expDir);
end
[obj.net, obj.stats] = CalvinNN.loadState(modelPath(targetEpoch));

% Replace softmaxloss layer with softmax layer
softmaxlossInput = obj.net.layers(obj.net.getLayerIndex('softmaxloss')).inputs{1};
obj.net.removeLayer('softmaxloss');
obj.net.addLayer('softmax', dagnn.SoftMax(), softmaxlossInput, 'scores', {});
softmaxIdx = obj.net.layers(obj.net.getLayerIndex('softmax')).outputIndexes;
assert(numel(softmaxIdx) == 1);

% Set datasetMode in imdb
datasetMode = 'test';
obj.imdb.setDatasetMode(datasetMode);
state.epoch = targetEpoch;
state.allBatchInds = obj.imdb.getAllBatchInds();

% Process the epoch
if numGpus <= 1
    obj.stats.(datasetMode) = CalvinNN.process_epoch(obj.net, state, obj.imdb, obj.nnOpts, datasetMode);
else
    % Jasper: Probably the multi-gpu mode does not work because of accumulateStats
    % savedNet = obj.net.saveobj();
    spmd
        net_ = obj.net; % dagnn.DagNN.loadobj(savedNet);
        stats_.(datasetMode) = CalvinNN.process_epoch(net_, state, obj.imdb, obj.nnOpts, datasetMode);
        % if labindex == 1, savedNet_ = net_.saveobj(); end
    end
    % obj.net = dagnn.DagNN.loadobj(savedNet_{1});
    stats__ = CalvinNN.accumulateStats(stats_);
    obj.stats.(datasetMode) = stats__.(datasetMode);
end

% The stats are the desired results 
results = obj.stats.(datasetMode);

% Set the extractStatsFn function to original training function
obj.nnOpts.extractStatsFn = tempExtractStatsFn;

end