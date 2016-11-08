function evalDetection(testName, imdb, stats, nnOpts)
% evalDetection(testName, imdb, stats, nnOpts)
%
% Evaluates the detection results on a PASCAL VOC dataset.
%
% Copyrights by Jasper Uijlings, 2015
% Modified by Holger Caesar, 2016

% Settings
global DATAopts;
compName = 'FRCN'; % Name under which files are saved in VOC folder

% Get test images
testIms = imdb.misc.testIms;

% get image sizes
testCount = length(testIms);
imSizesPath = fullfile(nnOpts.expDir, 'imSizes-test.mat');
if exist(imSizesPath, 'file')
    imSizesStruct = load(imSizesPath, 'imSizes');
    imSizes = imSizesStruct.imSizes;
else
    imSizes = nan(testCount, 2);
    for i = 1 : testCount %testCount : -1 : 1 %TODO: why were these in reverse order?
        im = imread(sprintf(DATAopts.imgpath, testIms{i}));
        imSizes(i, :) = size(im);
    end
    save(imSizesPath, 'imSizes');
end

for cI = 1 : 20
    %
    currBoxes = cell(testCount, 1);
    currScores = cell(testCount, 1);
    for i = 1 : testCount
        currBoxes{i}  = stats.results(i).boxes{cI + 1};
        currScores{i} = stats.results(i).scores{cI + 1};
    end
    
    [currBoxes,  fileIdx]  = Cell2Matrix(gather(currBoxes));
    [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
    assert(isequal(fileIdx, fileIdx2)); % Should be equal
    
    currFilenames = testIms(fileIdx);
    
    [~, sI] = sort(currScores, 'descend');
    currScores = currScores(sI);
    currBoxes  = currBoxes(sI,:);
    currFilenames = currFilenames(sI);
    
    % Use default script to compute detection performance
    [recall{cI}, prec{cI}, ap(cI,1), upperBound{cI}] = ...
        DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
        compName, 1, nnOpts.misc.overlapNms); %#ok<AGROW>
    ap(cI)
end

if isfield(stats.results(1), 'boxesRegressed')
    for cI = 1 : 20
        %
        currBoxes  = cell(testCount, 1);
        currScores = cell(testCount, 1);
        
        for i=1:testCount
            % Get regressed boxes and refit them to the image
            currBoxes{i} = stats.results(i).boxesRegressed{cI+1};
            currBoxes{i}(:,1) = max(currBoxes{i}(:, 1), 1);
            currBoxes{i}(:,2) = max(currBoxes{i}(:, 2), 1);
            currBoxes{i}(:,3) = min(currBoxes{i}(:, 3), imSizes(i, 2));
            currBoxes{i}(:,4) = min(currBoxes{i}(:, 4), imSizes(i, 1));
            
            currScores{i} = stats.results(i).scoresRegressed{cI+1};
        end
        
        [currBoxes,  fileIdx]  = Cell2Matrix(gather(currBoxes));
        [currScores, fileIdx2] = Cell2Matrix(gather(currScores));
        assert(isequal(fileIdx, fileIdx2)); % Should be equal
        
        currFilenames = testIms(fileIdx);
        
        [~, sI] = sort(currScores, 'descend');
        currScores = currScores(sI);
        currBoxes  = currBoxes(sI, :);
        currFilenames = currFilenames(sI);
        
        % Use default script to compute detection performance
        [recall{cI}, prec{cI}, apRegressed(cI,1), upperBound{cI}] = ...
            DetectionToPascalVOCFiles(testName, cI, currBoxes, currFilenames, currScores, ...
            compName, 1, nnOpts.misc.overlapNms); %#ok<AGROW>
    end
else
    apRegressed = nan;
end

% Print performance
fprintf('Per-class results (default): \n');
disp(ap);
fprintf('Per-class results (regressed): \n');
disp(apRegressed);
fprintf('mAP: %.2f%% (default), %.2f%% (regressed)\n', mean(ap)*100, mean(apRegressed)*100);

% Save results to disk
save([nnOpts.expDir, '/', 'resultsEpochFinalTest.mat'], 'ap', 'apRegressed');