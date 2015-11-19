function train(obj)
% train(obj)
%
% TODO: 
% - Currently this method doesn't allow for testing. Either change it or implement a different method for that. 
% - Currently we cannot change the learning rate after 13 epochs.

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
        spmd, gpuDevice(obj.nnOpts.gpus(labindex)), end
    end
    
    % Delete previous memory mapping files
    if exist(obj.nnOpts.memoryMapFile, 'file')
        delete(obj.nnOpts.memoryMapFile);
    end
elseif numGpus == 1,
    gpuDevice(obj.nnOpts.gpus);
end;

modelPath = @(ep) fullfile(obj.nnOpts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(obj.nnOpts.expDir, 'net-train.pdf');

start = obj.nnOpts.continue * CalvinNN.findLastCheckpoint(obj.nnOpts.expDir);
if start >= 1
    fprintf('resuming by loading epoch %d\n', start);
    [obj.net, obj.stats] = CalvinNN.loadState(modelPath(start));
end

for epoch=start+1:obj.nnOpts.numEpochs
    
    % train one epoch
    state.epoch = epoch;
    state.learningRate = obj.nnOpts.learningRate(min(epoch, numel(obj.nnOpts.learningRate)));
    
    % Do training and validation
    theSets = {'train', 'val'};
    for datasetModeI = 1:numel(theSets),
        datasetMode = theSets{datasetModeI};
        % Set datasetMode in imdb
        obj.imdb.setDatasetMode(datasetMode);
        state.allBatchInds = obj.imdb.getAllBatchInds();
        
        if numGpus <= 1
            obj.stats.(datasetMode)(epoch) = CalvinNN.process_epoch(obj.net, state, obj.imdb, obj.nnOpts, datasetMode);
        else
            savedNet = obj.net.saveobj();
            spmd
                net_ = dagnn.DagNN.loadobj(savedNet);
                stats_.(datasetMode) = CalvinNN.process_epoch(net_, state, obj.imdb, obj.nnOpts, datasetMode);
                if labindex == 1, savedNet_ = net_.saveobj(); end
            end
            obj.net = dagnn.DagNN.loadobj(savedNet_{1});
            stats__ = CalvinNN.accumulateStats(stats_);
            obj.stats.(datasetMode)(epoch) = stats__.(datasetMode);
        end
    end
    
    % save
    if ~obj.nnOpts.evaluateMode,
        CalvinNN.saveState(modelPath(epoch), obj.net, obj.stats);
    end
    
    figure(1); clf;
    values = [];
    leg = {};
    for s = {'train', 'val'}
        s = char(s); %#ok<FXSET>
        for f = setdiff(fieldnames(obj.stats.train)', {'num', 'time'})
            f = char(f); %#ok<FXSET>
            leg{end+1} = sprintf('%s (%s)', f, s); %#ok<AGROW>
            tmp = [obj.stats.(s).(f)];
            values(end+1,:) = tmp(1,:)'; %#ok<AGROW>
        end
    end
    subplot(1,2,1); plot(1:epoch, values');
    legend(leg{:}); xlabel('epoch'); ylabel('metric');
    subplot(1,2,2); semilogy(1:epoch, values');
    legend(leg{:}); xlabel('epoch'); ylabel('metric');
    grid on;
    drawnow;
    print(1, modelFigPath, '-dpdf');
end