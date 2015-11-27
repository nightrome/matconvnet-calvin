function init(obj, nnOpts)
% init(obj, nnOpts)

defnnOpts.expDir = fullfile('data', 'exp');
defnnOpts.continue = false;
defnnOpts.batchSize = 2;
defnnOpts.numSubBatches = 2;
defnnOpts.gpus = [];
defnnOpts.prefetch = false;
defnnOpts.numEpochs = 16;
defnnOpts.learningRate = 0.001;
defnnOpts.weightDecay = 0.0005;
defnnOpts.momentum = 0.9;
defnnOpts.derOutputs = {'objective', 1};
defnnOpts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin');
defnnOpts.extractStatsFn = @CalvinNN.extractStats;

% Merge input settings with default settings
defnnOptsFields = fields(defnnOpts);
for fieldIdx = 1 : numel(defnnOptsFields),
    fieldName = defnnOptsFields{fieldIdx};
    if ~isfield(nnOpts, fieldName),
        nnOpts.(fieldName) = defnnOpts.(fieldName);
    end
end

% Check settings
assert(~nnOpts.prefetch, 'Error: Prefetch is not supported in Matconvnet-Calvin!');

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
    
    % Assign GPUs
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