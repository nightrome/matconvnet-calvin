function train(obj)
% train(obj)
%
% Main training script used for training and validation.
% Only <= 1 GPUs supported.
% 
% Copyright by Holger Caesar, 2016

modelFigPath = fullfile(obj.nnOpts.expDir, 'net-train.pdf');

for epoch = obj.nnOpts.initEpoch + 1 : obj.nnOpts.numEpochs
    
    % Set epoch and it's learning rate
    state.epoch = epoch;
    state.learningRate = obj.nnOpts.learningRate(min(epoch, numel(obj.nnOpts.learningRate)));
    
    % Set the current epoch in imdb
    obj.imdb.initEpoch(epoch);
    
    % Do training and validation
    datasetModes = {'train', 'val'};
    for datasetModeIdx = 1 : numel(datasetModes)
        datasetMode = datasetModes{datasetModeIdx};
        
        % Set train/val mode (disable Dropout etc.)
        obj.imdb.setDatasetMode(datasetMode);
        if strcmp(datasetMode, 'train'),
            obj.net.mode = 'train';
        else % val
            obj.net.mode = 'test';
        end;
        state.allBatchInds = obj.imdb.getAllBatchInds();
        
        obj.stats.(datasetMode)(epoch) = obj.processEpoch(obj.net, state);
    end
    
    % Save current snapshot
    obj.saveState(obj.nnOpts.modelPath(epoch));
    
    % Plot statistics
    if obj.nnOpts.plotEval
        plotAccuracy = isfield(obj.stats.val, 'accuracy') && obj.nnOpts.plotAccuracy;
        obj.plotStats(1:epoch, obj.stats, plotAccuracy);
        
        drawnow;
        print(1, modelFigPath, '-dpdf'); %#ok<MCPRT>
    end
end