function roiPooling_test()
% roiPooling_test()
%
% Copyright by Holger Caesar, 2015

doTest = [true, false, true];

%%% Forward pass
%%% Test 1: multiple of the resolution
if doTest(1),
    test1();
end;

%%% Test 2: not a multiple of the resolution
if doTest(2),
    test2();
end;

%%% Backward pass
%%% Test 3: simple backprop
if doTest(3),
    test3();
end;


function test1()
clear;
% Input
onGpu = false;
channelCount = 2;
convIm = single(zeros(16, 8, channelCount));
convIm(1, 1, 1) = 1;
convIm(1, 2, 1) = 2;
convIm(2, 2, 1) = 3;
oriImSize = [16, 8, 3];
roiPoolSize = [7, 7];
boxSize = roiPoolSize + 1;
boxStart = [1, 1];
boxes = single([boxStart, boxStart + boxSize - 1]);
if onGpu,
    convIm = gpuArray(convIm);
end;

% Output
targetRois = zeros([roiPoolSize, channelCount], 'single');
targetRois(1, 1, 1) = 1;
targetRois(1, 2, 1) = 2;
targetRois(2, 2, 1) = 3;
targetMasks = nan([roiPoolSize, channelCount], 'single');
targetMasks(1, 1, 1) = sub2ind(size(convIm), 1, 1, 1) - 1;
targetMasks(1, 2, 1) = sub2ind(size(convIm), 1, 2, 1) - 1;
targetMasks(2, 2, 1) = sub2ind(size(convIm), 2, 2, 1) - 1;
if onGpu,
    targetRois = gpuArray(targetRois);
    targetMasks = gpuArray(convIm);
end;

% Run
[rois, masks] = roiPooling_forward(convIm, oriImSize, boxes, roiPoolSize);

% Check
assert(isequal(targetRois, rois));
validMask = ~isnan(targetMasks(:));
assert(isequal(targetMasks(validMask), masks(validMask)));
fprintf('Forward pass: success!\n');

function test2()
clear;
% Input
onGpu = false;
channelCount = 2;
convIm = single(zeros(16, 8, channelCount));
convIm(1, 1, 1) = 1;
convIm(1, 2, 1) = 2;
convIm(2, 2, 1) = 3;
oriImSize = [16, 8, 3];
roiPoolSize = [7, 7];
boxes = single([1, 1, 4, 4]);
if onGpu,
    convIm = gpuArray(convIm);
end;

error('TODO: Define target!');

% Output
targetRois = zeros([roiPoolSize, channelCount], 'single');
targetRois(1, 1, 1) = 1;
targetRois(1, 2, 1) = 2;
targetRois(2, 2, 1) = 3;
targetMasks = nan([roiPoolSize, channelCount], 'single');
targetMasks(1, 1, 1) = sub2ind(size(convIm), 1, 1, 1) - 1;
targetMasks(1, 2, 1) = sub2ind(size(convIm), 1, 2, 1) - 1;
targetMasks(2, 2, 1) = sub2ind(size(convIm), 2, 2, 1) - 1;
if onGpu,
    targetRois = gpuArray(targetRois);
    targetMasks = gpuArray(convIm);
end;

% Run
[rois, masks] = roiPooling_forward(convIm, oriImSize, boxes, roiPoolSize);

% Check
assert(isequal(targetRois, rois));
validMask = ~isnan(targetMasks(:));
assert(isequal(targetMasks(validMask), masks(validMask)));
fprintf('Forward pass: success!\n');

function test3()
clear;
% Input
boxCount = 2;
channelCount = 2;
convImSize = [100, 100, channelCount];
roiPoolSize = [7, 7];
masks = nan([roiPoolSize, channelCount, boxCount], 'single');
masks(8) = 5;
dzdx = rand(size(masks), 'single');

% Output
targetDzdxout = zeros(convImSize, 'single');
targetDzdxout(5+1) = dzdx(8);

% Run
dzdxout = roiPooling_backward(boxCount, convImSize, roiPoolSize, masks, dzdx);

% Check
assert(isequal(targetDzdxout, dzdxout));
fprintf('Backward pass: success!\n');