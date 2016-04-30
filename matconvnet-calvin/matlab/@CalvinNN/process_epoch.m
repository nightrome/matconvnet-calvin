function stats = process_epoch(obj, net, state)
% stats = process_epoch(obj, net, state)
%
% Processes one training/validation epoch.
%
% Copyright by Matconvnet
% Modified by Holger Caesar, 2016

% Initialize momentum on this worker
if strcmp(obj.imdb.datasetMode, 'train')
    state.momentum = num2cell(zeros(1, numel(net.params)));
end

% Move data to GPU and create memory map file for multiple GPUs
numGpus = numel(obj.nnOpts.gpus);
assert(numGpus <= 1);
if numGpus == 1
    net.move('gpu');
    if strcmp(obj.imdb.datasetMode, 'train')
        state.momentum = cellfun(@gpuArray, state.momentum, 'UniformOutput', false);
    end
end

% Get the indices of all batches
allBatchInds = state.allBatchInds;
assert(~isempty(allBatchInds));

% Initialize
start = tic;
num = 0;
targetTime = 3; % Counter to not pollute screen with printf statements
waitTime = 10;  % How long to wait to update the fprintf statements.
fprintf('Starting epoch: %d\n', state.epoch);

for t=1:obj.nnOpts.batchSize:numel(allBatchInds),
    batchNumElements = 0;
    
    for s=1:obj.nnOpts.numSubBatches,
        % get this image batch and prefetch the next
        batchStart = t + (s-1);
        batchEnd = min(t+obj.nnOpts.batchSize-1, numel(allBatchInds));
        batchInds = allBatchInds(batchStart : obj.nnOpts.numSubBatches : batchEnd);
        num = num + numel(batchInds);
        if numel(batchInds) == 0, continue; end
        
        [inputs, numElements] = obj.imdb.getBatch(batchInds, net, obj.nnOpts);
        % Skip empty subbatches
        if numElements == 0
            continue;
        end
        
        if strcmp(obj.imdb.datasetMode, 'train')
            net.accumulateParamDers = (s ~= 1);
            net.eval(inputs, obj.nnOpts.derOutputs);
        else
            net.eval(inputs);
        end
        
        % Extract results at test time (not val)
        if strcmp(obj.imdb.datasetMode, 'test')
            % Pre-allocate the struct-array for the results
            currResult = obj.nnOpts.testFn(obj.imdb, obj.nnOpts, net, inputs, batchInds);
            if t == 1 && s == 1
                results = repmat(currResult(1), numel(allBatchInds), 1);
            end
            results(batchInds) = currResult(1:numel(batchInds));
        end
        
        batchNumElements = batchNumElements + numElements;
    end
    
    % Extract learning stats
    if ~strcmp(obj.imdb.datasetMode, 'test')
        stats = obj.nnOpts.extractStatsFn(net, inputs);
    end
    
    % Accumulate gradients
    if strcmp(obj.imdb.datasetMode, 'train')
        state = obj.accumulate_gradients(state, net, batchNumElements);
    end
    
    % Print learning statistics
    stats.num = num;
    stats.time = toc(start);
    
    if stats.time > targetTime || numel(allBatchInds) <= obj.nnOpts.batchSize
        targetTime = targetTime + waitTime;
        fprintf('%s: epoch %02d: %3d/%3d: %.1f Hz', ...
            obj.imdb.datasetMode, ...
            state.epoch, ...
            fix(t/obj.nnOpts.batchSize)+1, ceil(numel(allBatchInds)/obj.nnOpts.batchSize), ...
            stats.num/stats.time);
        for field = setdiff(fieldnames(stats)', {'num', 'time', 'results'}) 
            field = char(field); %#ok<FXSET>
            fprintf(' %s:', field);
            fprintf(' %.3f', stats.(field));
        end
        fprintf('\n');
    end
end

% Give back results at test time
if strcmp(obj.imdb.datasetMode, 'test')
    stats.results = results; 
end

fprintf('Finished!\n');

net.reset();
net.move('cpu');