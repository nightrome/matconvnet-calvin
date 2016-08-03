function setupFastRcnnRegions(varargin)
% setupFastRcnnRegions(varargin)
%
% Extract Selective Search proposals and ground-truth for each image in the
% PASCAL VOC 20xx dataset. Note that this takes about 4s/im or about 11h
% for VOC 2010.
%
% Copyright by Holger Caesar, 2016

% Settings
trainName = 'train';
testName = 'val';

% Get image lists
global DATAopts;
trainIms = textread(sprintf(DATAopts.imgsetpath, trainName), '%s'); %#ok<DTXTRD>
testIms  = textread(sprintf(DATAopts.imgsetpath, testName), '%s'); %#ok<DTXTRD>

% Check if regions were already extracted by looking at the last file
lastFile = [DATAopts.gStructPath, testIms{end}, '.mat'];
if ~exist(lastFile, 'file')
    % Create output folder
    if ~exist(DATAopts.gStructPath, 'dir')
        mkdir(DATAopts.gStructPath);
    end
    
    for idxImg = 1:size(trainIms,1)
        fprintf('Processing train img: %d/%d\n', idxImg, size(trainIms, 1));
        boxPath = [DATAopts.gStructPath, trainIms{idxImg}, '.mat'];
        if exist(boxPath, 'file')
            fprintf('Skipping existing file: %s\n', boxPath);
            continue;
        end
        boxStruct = GetGTAndSSBoxes(trainIms{idxImg}); %#ok<NASGU>
        save(boxPath, '-struct', 'boxStruct');
    end
    
    for idxImg = 1:size(testIms,1)
        fprintf('Processing test img: %d/%d\n', idxImg, size(testIms, 1));
        boxPath = [DATAopts.gStructPath, testIms{idxImg}, '.mat'];
        if exist(boxPath, 'file')
            fprintf('Skipping existing file: %s\n', boxPath);
            continue;
        end
        boxStruct = GetGTAndSSBoxes(testIms{idxImg}); %#ok<NASGU>
        save(boxPath, '-struct', 'boxStruct');
    end
end