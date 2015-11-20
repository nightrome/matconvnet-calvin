function stats = process_epoch(obj, net, state)
% stats = process_epoch(obj, net, state)
%
% Note that net needs to be a separate argument (not obj.net) to support
% multiple GPUs.

% Check options
assert(~obj.nnOpts.prefetch, 'Error: Prefetch is not supported in Matconvnet-Calvin!');

if strcmp(obj.imdb.datasetMode, 'train')
    state.momentum = num2cell(zeros(1, numel(net.params)));
end

numGpus = numel(obj.nnOpts.gpus);
if numGpus >= 1
    net.move('gpu');
    if strcmp(obj.imdb.datasetMode, 'train')
        state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);
    end
end
if numGpus > 1
    mmap = obj.map_gradients(net);
else
    mmap = [];
end

stats.time = 0;
stats.scores = [];
allBatchInds = state.allBatchInds;
assert(~isempty(allBatchInds));
start = tic;
num = 0;

for t=1:obj.nnOpts.batchSize:numel(allBatchInds),
    batchNumElements = 0;
    
    for s=1:obj.nnOpts.numSubBatches,
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs;
        batchEnd = min(t+obj.nnOpts.batchSize-1, numel(allBatchInds));
        batchInds = allBatchInds(batchStart : obj.nnOpts.numSubBatches * numlabs : batchEnd);
        num = num + numel(batchInds);
        if numel(batchInds) == 0, continue; end
        
        [inputs, numElements] = obj.imdb.getBatch(batchInds, net);
        % Skip empty subbatches
        if numElements == 0,
            continue;
        end
        
        if strcmp(obj.imdb.datasetMode, 'train')
            net.accumulateParamDers = (s ~= 1);
            net.eval(inputs, obj.nnOpts.derOutputs);
        else
            net.eval(inputs);
        end
        
        batchNumElements = batchNumElements + numElements;
    end
    
    % extract learning stats
    stats = obj.nnOpts.extractStatsFn(net);
    
    % accumulate gradient
    if strcmp(obj.imdb.datasetMode, 'train')
        if ~isempty(mmap)
            obj.write_gradients(mmap, net);
            labBarrier();
        end
        state = obj.accumulate_gradients(state, net, batchNumElements, mmap);
    end
    
    % print learning statistics
    stats.num = num;
    stats.time = toc(start);
    
    fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
        obj.imdb.datasetMode, ...
        state.epoch, ...
        fix(t/obj.nnOpts.batchSize)+1, ceil(numel(allBatchInds)/obj.nnOpts.batchSize), ...
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