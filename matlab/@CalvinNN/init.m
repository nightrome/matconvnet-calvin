function init(obj, varargin)
% init(obj, varargin)

nnOpts.expDir = fullfile('data','exp') ;
nnOpts.continue = false ;
nnOpts.batchSize = 256 ;
nnOpts.numSubBatches = 1 ;
nnOpts.gpus = [] ;
nnOpts.prefetch = false ;
nnOpts.numEpochs = 300 ;
nnOpts.learningRate = 0.001 ;
nnOpts.weightDecay = 0.0005 ;
nnOpts.momentum = 0.9 ;
nnOpts.derOutputs = {'objective', 1} ;
nnOpts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
nnOpts.extractStatsFn = @extractStats ;

% Matconvnet-calvin options
nnOpts.netPath = '';
nnOpts.roiPool.use = true;
nnOpts.roiPool.roiPoolFreeform = false;

nnOpts = vl_argparse(nnOpts, varargin);

if ~exist(nnOpts.expDir, 'dir'), mkdir(nnOpts.expDir) ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

% setup GPUs
numGpus = numel(nnOpts.gpus) ;
if numGpus > 1
    if isempty(gcp('nocreate')),
        parpool('local',numGpus) ;
        spmd, gpuDevice(nnOpts.gpus(labindex)), end
    end
    if exist(nnOpts.memoryMapFile, 'file')
        delete(nnOpts.memoryMapFile) ;
    end
elseif numGpus == 1
    gpuDevice(nnOpts.gpus)
end

% Set new fields
obj.nnOpts = nnOpts;