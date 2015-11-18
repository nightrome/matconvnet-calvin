function train(obj)
% train(obj)

% setup GPUs
numGpus = numel(obj.nnOpts.gpus);
if numGpus > 1,
    if isempty(gcp('nocreate')),
        parpool('local',numGpus);
        spmd, gpuDevice(obj.nnOpts.gpus(labindex)), end
    end
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
    [obj.net, obj.stats] = loadState(modelPath(start));
end

for epoch=start+1:obj.nnOpts.numEpochs
    
    % train one epoch
    state.epoch = epoch;
    state.learningRate = obj.nnOpts.learningRate(min(epoch, numel(obj.nnOpts.learningRate)));
    state.train = randperm(numel(obj.imdb.getNumBatches('train'))); % shuffle
    state.val = 1:numel(obj.imdb.getNumBatches('val'));
    
    if numGpus <= 1
        obj.stats.train(epoch) = CalvinNN.process_epoch(obj.net, state, obj.imdb, obj.nnOpts, 'train');
        obj.stats.val(epoch) = CalvinNN.process_epoch(obj.net, state, obj.imdb, obj.nnOpts, 'val');
    else
        savedNet = obj.net.saveobj();
        spmd
            net_ = dagnn.DagNN.loadobj(savedNet);
            stats_.train = CalvinNN.process_epoch(net_, state, obj.imdb, obj.nnOpts, 'train');
            stats_.val = CalvinNN.process_epoch(net_, state, obj.imdb, obj.nnOpts, 'val');
            if labindex == 1, savedNet_ = net_.saveobj(); end
        end
        obj.net = dagnn.DagNN.loadobj(savedNet_{1});
        stats__ = CalvinNN.accumulateStats(stats_);
        obj.stats.train(epoch) = stats__.train;
        obj.stats.val(epoch) = stats__.val;
    end
    
    % save
    if ~obj.nnOpts.evaluateMode,
        CalvinNN.saveState(modelPath(epoch), obj.net, obj.stats);
    end
    
    figure(1); clf;
    values = [];
    leg = {};
    for s = {'train', 'val'}
        s = char(s);
        for f = setdiff(fieldnames(obj.stats.train)', {'num', 'time'})
            f = char(f);
            leg{end+1} = sprintf('%s (%s)', f, s);
            tmp = [obj.stats.(s).(f)];
            values(end+1,:) = tmp(1,:)';
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