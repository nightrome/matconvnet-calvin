function results = testDetection(imdb, nnOpts, net, inputs)
% Get predicted boxes and scores per class
% Only gets top nnOpts.maxNumBoxesPerImTest boxes (default: 5000)
% Only gets boxes with score higher than nnOpts.minDetectionScore (default: 0.01)
% NMS threshold: nnOpts.nmsTTest (default: 0.3)

% Variables which should probably be in imdb.nnOpts or something
% Jasper: Probably need to do something more robust here
if isfield(nnOpts, 'maxNumBoxesPerImTest')
    maxNumBoxesPerImTest = nnOpts.maxNumBoxesPerImTest;
else
    maxNumBoxesPerImTest = 5000;
end

if isfield(nnOpts, 'nmsTTest')
    nmsTTest = imdb.nmsTTest;
else
    nmsTTest = 0.3; % non-maximum threshold
end

if isfield(nnOpts, 'minDetectionScore')
    minDetectionScore = nnOpts.minDetectionScore;
else
    minDetectionScore = 0.01;
end

% Get scores
vI = net.getVarIndex('scores');
scoresStruct = net.vars(vI);
scores = permute(scoresStruct.value, [4 3 2 1]);

% Get boxes
inputNames = inputs(1:2:end);
[~, boxI] = ismember('boxes', inputNames);
boxI = boxI * 2; % Index of actual argument
boxes = inputs{boxI};

% Get regression targets for boxes
if imdb.boxRegress
    vI = net.getVarIndex('xRegress');
    regressStruct = net.vars(vI);
    regressFactors = permute(regressStruct.value, [4 3 2 1]);
else
    regressFactors = zeros(size(boxes));
end

% Get top boxes for each category. Perform NMS. Thresholds defined at top of function
currMaxBoxes = min(maxNumBoxesPerImTest, size(boxes, 1));
for cI = size(scores,2):-1:1
    % Get top scores and boxes
    [currScores, sI] = sort(scores(:,cI), 'descend');
    currScores = currScores(1:currMaxBoxes);
    sI = sI(1:currMaxBoxes);
    currBoxes = boxes(sI,:);
    
    % Do regression
    currRegressF = regressFactors(sI,:);
    currBoxesReg = BoxRegresss(currBoxes, currRegressF);
    
    % Get scores (w boxes) above certain threshold
    goodI = currScores > minDetectionScore;
    currScores = currScores(goodI,:);
    currBoxes = currBoxes(goodI,:);
    currBoxesReg = currBoxesReg(goodI,:);
    
    % Perform NMS
    [~, goodBoxesI] = BoxNMS(currBoxes, nmsTTest);
    currBoxes = currBoxes(goodBoxesI,:);
    currScores = currScores(goodBoxesI,:);
    currBoxesReg = currBoxesReg(goodBoxesI,:);
    
    results.boxes{cI} = gather(currBoxes);
    results.scores{cI} = gather(currScores);
    
    if imdb.boxRegress
        results.regress{cI} = currBoxesReg;
    end
end