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
        plotAccuracy = isfield(obj.stats.val, 'accuracy') && obj.nnOpts.plotAccuracy;
        cmap = parula(3);
        cmap(3, :) = [255, 140, 0] / 255;
        
        if true
            figure(1); clf;
            if plotAccuracy
                subplot(2, 1, 1);
            end
            hold on;
            leg = {};
            datasetModes = {'train', 'val'};
            for datasetModeIdx = 1 : numel(datasetModes)
                datasetMode = datasetModes{datasetModeIdx};
                field = 'objective';
                fieldValues = [obj.stats.(datasetMode).(field)];
                if datasetModeIdx == 1
                    marker = '-';
                else
                    marker = '--';
                end
                
                % For i.e. objective with 1 value
                leg{end + 1} = sprintf('%s (%s)', field, datasetMode); %#ok<AGROW>
                values = fieldValues(1, :)';
                plot(1:epoch, values, 'Color', cmap(1, :), 'LineStyle', marker);
            end
            legend(leg);
            xlabel('epoch');
            ylabel('objective');
            grid on;
        end
        
        if plotAccuracy
            subplot(2, 1, 2);
            hold on;
            
            leg = {};
            datasetModes = {'train', 'val'};
            for datasetModeIdx = 1 : numel(datasetModes)
                datasetMode = datasetModes{datasetModeIdx};
                field = 'accuracy';
                fieldValues = [obj.stats.(datasetMode).(field)];
                if datasetModeIdx == 1
                    marker = '-';
                else
                    marker = '--';
                end
                
                leg{end + 1} = sprintf('Pix. Acc. (%s)', datasetMode); %#ok<AGROW>
                values = fieldValues(1, :)';
                plot(1:epoch, values, 'Color', cmap(1, :), 'LineStyle', marker);

                leg{end + 1} = sprintf('Class. Acc. (%s)', datasetMode); %#ok<AGROW>
                values = fieldValues(2, :)';
                plot(1:epoch, values, 'Color', cmap(2, :), 'LineStyle', marker);
                
                leg{end + 1} = sprintf('Mean IU (%s)', datasetMode); %#ok<AGROW>
                values = fieldValues(3, :)';
                plot(1:epoch, values, 'Color', cmap(3, :), 'LineStyle', marker);
            end
            legend(leg);
            xlabel('epoch');
            ylabel('accuracy');
            grid on;
        end
        
        drawnow;
        print(1, modelFigPath, '-dpdf'); %#ok<MCPRT>
    end
end