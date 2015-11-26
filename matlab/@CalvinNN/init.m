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

% Matconvnet-calvin options
defnnOpts.evaluateMode = false;

% Merge settings
defnnOptsFields = fields(defnnOpts);
for fieldIdx = 1 : numel(defnnOptsFields),
    fieldName = defnnOptsFields{fieldIdx};
    if ~isfield(nnOpts, fieldName),
        nnOpts.(fieldName) = defnnOpts.(fieldName);
    end
end

% Do not create directory in evaluation mode
if ~nnOpts.evaluateMode && ~exist(nnOpts.expDir, 'dir'),
    mkdir(nnOpts.expDir);
end

% Set new fields
obj.nnOpts = nnOpts;