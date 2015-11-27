function train(obj)
% train(obj)
%
% TODO: 
% - Currently this method doesn't allow for testing. Either change it or implement a different method for that. 
% - Currently we cannot change the learning rate after 13 epochs.
% - Only <=2 GPUs at the same time seem to work

modelPath = @(ep) fullfile(obj.nnOpts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(obj.nnOpts.expDir, 'net-train.pdf');
numGpus = numel(obj.nnOpts.gpus);

% Load previous training snapshot
start = obj.nnOpts.continue * CalvinNN.findLastCheckpoint(obj.nnOpts.expDir);
if start >= 1
    fprintf('Resuming by loading epoch %d\n', start);
    [obj.net, obj.stats] = CalvinNN.loadState(modelPath(start));
end

% Make sure the net is set to training mode ('normal' means dropout is active)
obj.net.mode = 'train';

for epoch=start+1:obj.nnOpts.numEpochs
    
    % train one epoch
    state.epoch = epoch;
    state.learningRate = obj.nnOpts.learningRate(min(epoch, numel(obj.nnOpts.learningRate)));
    
    obj.imdb.switchFlipLR();
    
    % Do training and validation
    datasetModes = {'train', 'val'};
    for datasetModeIdx = 1:numel(datasetModes)
        datasetMode = datasetModes{datasetModeIdx};
        
        % Set datasetMode in imdb
        obj.imdb.setDatasetMode(datasetMode);
        state.allBatchInds = obj.imdb.getAllBatchInds();
        
        if numGpus <= 1
            obj.stats.(datasetMode)(epoch) = obj.process_epoch(obj.net, state);
        else
            savedNet = obj.net.saveobj();
            spmd
                net_ = dagnn.DagNN.loadobj(savedNet);
                stats_.(datasetMode) = obj.process_epoch(net_, state);
                if labindex == 1, savedNet_ = net_.saveobj(); end
            end
            obj.net = dagnn.DagNN.loadobj(savedNet_{1});
            stats__ = obj.accumulateStats(stats_);
            obj.stats.(datasetMode)(epoch) = stats__.(datasetMode);
        end
    end
    
    % Save current snapshot
    obj.saveState(modelPath(epoch));
    
    % Plot statistics
    figure(1); clf;
    values = [];
    leg = {};
    datasetModes = {'train', 'val'};
    for datasetModeIdx = 1:numel(datasetModes)
        datasetMode = datasetModes{datasetModeIdx};
        
        for f = setdiff(fieldnames(obj.stats.train)', {'num', 'time'})
            f = char(f); %#ok<FXSET>
            leg{end+1} = sprintf('%s (%s)', f, datasetMode); %#ok<AGROW>
            tmp = [obj.stats.(datasetMode).(f)];
            values(end+1,:) = tmp(1,:)'; %#ok<AGROW>
        end
    end
    plot(1:epoch, values');
    legend(leg{:}); xlabel('epoch'); ylabel('metric');
    grid on;
    drawnow;
    print(1, modelFigPath, '-dpdf');
end