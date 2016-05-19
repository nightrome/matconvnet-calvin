function train(obj)
% train(obj)
%
% Main training script used for training and validation.
% Only <= 1 GPUs supported.
% 
% Copyright by Holger Caesar, 2016

modelPath = @(ep) fullfile(obj.nnOpts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(obj.nnOpts.expDir, 'net-train.pdf');
numGpus = numel(obj.nnOpts.gpus);
assert(numGpus <= 1);

% Load previous training snapshot
lastCheckPoint = CalvinNN.findLastCheckpoint(obj.nnOpts.expDir);
if isnan(lastCheckPoint)
    % No checkpoint found
    start = 0;
    
    % Save untrained net
    obj.saveState(modelPath(start));
else
    % Load existing checkpoint and continue
    start = obj.nnOpts.continue * lastCheckPoint;
    fprintf('Resuming by loading epoch %d\n', start);
    if start >= 1
        [obj.net, obj.stats] = CalvinNN.loadState(modelPath(start));
    end
end

for epoch = start + 1 : obj.nnOpts.numEpochs
    
    % Set epoch and it's learning rate
    state.epoch = epoch;
    state.learningRate = obj.nnOpts.learningRate(min(epoch, numel(obj.nnOpts.learningRate)));
    
    % Set the current epoch in imdb
    obj.imdb.initEpoch(epoch);
    
    % Do training and validation
    datasetModes = {'train', 'val'};
    for datasetModeIdx = 1:numel(datasetModes)
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
    obj.saveState(modelPath(epoch));
    
    % Plot statistics
    if obj.nnOpts.plotEval
        figure(1); clf;
        values = [];
        leg = {};
        datasetModes = {'train', 'val'};
        for datasetModeIdx = 1 : numel(datasetModes)
            datasetMode = datasetModes{datasetModeIdx};
            fields = setdiff(fieldnames(obj.stats.train), {'num', 'time'});
            
            for fieldIdx = 1 : numel(fields)
                field = fields{fieldIdx};
                fieldValues = [obj.stats.(datasetMode).(field)];
                
                for i = 1 : size(fieldValues, 1)
                    if size(fieldValues, 1) == 1 || ~nnOpts.plotEvalAll
                        % For i.e. objective with 1 value
                        leg{end + 1} = sprintf('%s (%s)', field, datasetMode); %#ok<AGROW>
                    else
                        % For i.e. accuracy with 3 values
                        leg{end + 1} = sprintf('%s-%d (%s)', field, i, datasetMode); %#ok<AGROW>
                    end
                    values(end + 1, :) = fieldValues(i, :)'; %#ok<AGROW>
                end
            end
        end
        plot(1:epoch, values');
        legend(leg{:});
        xlabel('epoch');
        ylabel('objective');
        grid on;
        drawnow;
        print(1, modelFigPath, '-dpdf'); %#ok<MCPRT>
    end
end