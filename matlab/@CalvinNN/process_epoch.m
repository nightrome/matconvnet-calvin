function stats = process_epoch(net, state, imdb, nnOpts, mode)
% stats = process_epoch(net, state, imdb, nnOpts, mode)

% Check options
assert(~nnOpts.prefetch, 'Error: Prefetch is not supported in Matconvnet-Calvin!');

if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params)));
end

numGpus = numel(nnOpts.gpus);
if numGpus >= 1
    net.move('gpu');
    if strcmp(mode,'train')
        sate.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false);
    end
end
if numGpus > 1
    mmap = map_gradients(nnOpts.memoryMapFile, net, numGpus);
else
    mmap = [];
end

stats.time = 0;
stats.scores = [];
subset = imdb.(mode);
assert(~isempty(subset));
start = tic;
num = 0;

for t=1:nnOpts.batchSize:numel(subset),
    batchSize = min(nnOpts.batchSize, numel(subset) - t + 1);
    
    for s=1:nnOpts.numSubBatches,
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs;
        batchEnd = min(t+nnOpts.batchSize-1, numel(subset));
        batch = subset(batchStart : nnOpts.numSubBatches * numlabs : batchEnd);
        num = num + numel(batch);
        if numel(batch) == 0, continue; end
        
        inputs = imdb.getBatch(imdb, net, nnOpts, batch);
        
        if strcmp(mode, 'train')
            net.accumulateParamDers = (s ~= 1);
            net.eval(inputs, nnOpts.derOutputs);
        else
            net.eval(inputs);
        end
    end
    
    % extract learning stats
    stats = nnOpts.extractStatsFn(net);
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            CalvinNN.write_gradients(mmap, net);
            labBarrier();
        end
        state = CalvinNN.accumulate_gradients(state, net, nnOpts, batchSize, mmap);
    end
    
    % print learning statistics
    stats.num = num;
    stats.time = toc(start);
    
    fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
        mode, ...
        state.epoch, ...
        fix(t/nnOpts.batchSize)+1, ceil(numel(subset)/nnOpts.batchSize), ...
        stats.num/stats.time * max(numGpus, 1));
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f);
        fprintf(' %s:', f);
        fprintf(' %.3f', stats.(f));
    end
    fprintf('\n');
end

net.reset();
net.move('cpu');