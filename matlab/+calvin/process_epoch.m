function stats = process_epoch(net, state, imdb, opts, mode)
% stats = process_epoch(net, state, imdb, opts, mode)

% Check options
assrt(~opts.prefetch, 'Error: Prefetch is not suppoted in Matconvnet-Calvin!');

if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        sate.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

stats.time = 0 ;
stats.scores = [] ;
subset = state.(mode) ;
start = tic ;
num = 0 ;

for t=1:opts.batchSize:numel(subset)
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        inputs = imdb.getBatch(batch) ;
        
        if strcmp(mode, 'train')
            net.accumulateParamDers = (s ~= 1) ;
            net.eval(inputs, opts.derOutputs) ;
        else
            net.eval(inputs) ;
        end
    end
    
    % extract learning stats
    stats = opts.extractStatsFn(net) ;
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
    end
    
    % print learning statistics
    time = toc(start) ;
    stats.num = num ;
    stats.time = toc(start) ;
    
    fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
        mode, ...
        state.epoch, ...
        fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize), ...
        stats.num/stats.time * max(numGpus, 1)) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.3f', stats.(f)) ;
    end
    fprintf('\n') ;
end

net.reset() ;
net.move('cpu') ;