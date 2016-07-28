function setupFastRcnnRegions(varargin)
% setupFastRcnnRegions(varargin)
%
% Extract Selective Search proposals and ground-truth for each image in the
% PASCAL VOC 20xx dataset.
%
% Copyright by Holger Caesar, 2016

trainName = 'train';
testName = 'val';
global DATAopts;
trainIms = textread(sprintf(DATAopts.imgsetpath, trainName), '%s'); %#ok<DTXTRD>
testIms = textread(sprintf(DATAopts.imgsetpath, testName), '%s'); %#ok<DTXTRD>


for idxImg = 1:size(trainIms,1)
    fprintf('Processing img: %d/%d\n', idxImg, size(trainIms, 1));
    boxStruct = GetGTAndSSBoxes(trainIms{idxImg}); %#ok<NASGU>
    save([DATAopts.gStructPath, trainIms{idxImg} '.mat'], '-struct', 'boxStruct'); 
end

for idxImg = 1:size(testIms,1)
    fprintf('Processing img: %d/%d\n', idxImg, size(testIms, 1));
    boxStruct = GetGTAndSSBoxes(testIms{idxImg}); %#ok<NASGU>
    save([DATAopts.gStructPath, testIms{idxImg} '.mat'], '-struct', 'boxStruct'); 
end