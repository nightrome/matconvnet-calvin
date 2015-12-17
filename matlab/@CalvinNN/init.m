function init(obj, varargin)
% init(obj, nnOpts)

defnnOpts.expDir = fullfile('data', 'exp');
defnnOpts.continue = false;
defnnOpts.batchSize = 2;
defnnOpts.numSubBatches = 2;
defnnOpts.gpus = [];
defnnOpts.prefetch = false;
defnnOpts.numEpochs = 16;
defnnOpts.learningRate = [repmat(1e-3, [1, 12]), repmat(1e-4, [1, 4])];
defnnOpts.weightDecay = 0.0005;
defnnOpts.momentum = 0.9;
defnnOpts.derOutputs = {'objective', 1};
defnnOpts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin');
defnnOpts.extractStatsFn = @CalvinNN.extractStats;
defnnOpts.testFn = @(imdb, nnOpts, net, inputs, batchInds) error('Error: Test function not implemented'); % function used at test time to evaluate performance
defnnOpts.misc = struct(); % fields used by custom layers are stored here

% Fast R-CNN options
defnnOpts.fastRcnn = true;
defnnOpts.bboxRegress = true;
defnnOpts.misc.roiPool.use = true;
defnnOpts.misc.roiPool.freeform.use = false;

% Merge input settings with default settings
nnOpts = vl_argparse(defnnOpts, varargin, 'nonrecursive');

% Check settings
assert(~nnOpts.prefetch, 'Error: Prefetch is not supported in Matconvnet-Calvin!');
assert(numel(nnOpts.learningRate) == 1 || numel(nnOpts.learningRate) == nnOpts.numEpochs);

% Do not create directory in evaluation mode
if ~exist(nnOpts.expDir, 'dir') && ~isempty(nnOpts.expDir),
    mkdir(nnOpts.expDir);
end

% Setup GPUs and memory map file
numGpus = numel(nnOpts.gpus);
if numGpus > 1,
    pool = gcp('nocreate');
    
    % Delete parpool with wrong size
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(gcp);
    end
    
    % Create new parpool
    if isempty(pool) || ~pool.isvalid()
        parpool('local', numGpus);
    end
    
    % Assign GPUs to SPMD workers
    spmd,
        gpuDevice(nnOpts.gpus(labindex))
    end
    
    % Delete previous memory mapping files
    if exist(nnOpts.memoryMapFile, 'file')
        delete(nnOpts.memoryMapFile);
    end
elseif numGpus == 1,
    gpuDevice(nnOpts.gpus);
end

% Set new fields
obj.nnOpts = nnOpts;