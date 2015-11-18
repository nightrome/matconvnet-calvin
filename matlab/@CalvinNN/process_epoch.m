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
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false);
    end
end
if numGpus > 1
    mmap = map_gradients(nnOpts.memoryMapFile, net, numGpus);
else
    mmap = [];
end

stats.time = 0;
stats.scores = [];
allBatchInds = state.allBatchInds;
assert(~isempty(allBatchInds));
start = tic;
num = 0;

for t=1:nnOpts.batchSize:numel(allBatchInds),
    %TODO: can be removed!
    %     batchSize = min(nnOpts.batchSize, numel(allBatchInds) - t + 1);
    batchNumElements = 0;
    
    for s=1:nnOpts.numSubBatches,
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs;
        batchEnd = min(t+nnOpts.batchSize-1, numel(allBatchInds));
        batchInds = allBatchInds(batchStart : nnOpts.numSubBatches * numlabs : batchEnd);
        num = num + numel(batchInds);
        if numel(batchInds) == 0, continue; end
        
        [inputs, numElements] = imdb.getBatch(batchInds, net);
        % Skip empty subbatches
        if numElements == 0,
            continue;
        end;
        
        if strcmp(mode, 'train')
            net.accumulateParamDers = (s ~= 1);
            net.eval(inputs, nnOpts.derOutputs);
        else
            net.eval(inputs);
        end
        
        batchNumElements = batchNumElements + numElements;
    end
    
    % extract learning stats
    stats = nnOpts.extractStatsFn(net);
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            CalvinNN.write_gradients(mmap, net);
            labBarrier();
        end
        state = CalvinNN.accumulate_gradients(state, net, nnOpts, batchNumElements, mmap);
    end
    
    % print learning statistics
    stats.num = num;
    stats.time = toc(start);
    
    fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
        mode, ...
        state.epoch, ...
        fix(t/nnOpts.batchSize)+1, ceil(numel(allBatchInds)/nnOpts.batchSize), ...
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