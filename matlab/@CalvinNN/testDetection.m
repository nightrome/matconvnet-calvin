function results = testDetection(imdb, nnOpts, net, inputs)
% Get predicted boxes and scores per class
% Only gets top nnOpts.maxNumBoxesPerImTest boxes (default: 50)
% NMS threshold: nnOpts.nmsTTest (default: 0.3)

% Variables which should probably be in imdb.nnOpts or something
% Jasper: Probably need to do something more robust here
if isfield(nnOpts, 'maxNumBoxesPerImTest')
    maxNumBoxesPerImTest = nnOpts.maxNumBoxesPerImTest;
else
    maxNumBoxesPerImTest = 50;
end

if isfield(nnOpts, 'nmsTTest')
    nmsTTest = imdb.nmsTTest;
else
    nmsTTest = 0.3; % non-maximum threshold
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

% Get top boxes for each category. Perform NMS. Thresholds defined at top of function
currMaxBoxes = min(maxNumBoxesPerImTest, size(boxes, 1));
for cI = size(scores,2):-1:1
    [currScores, sI] = sort(scores(:,cI), 'descend');
    currScores = currScores(1:currMaxBoxes);
    sI = sI(1:currMaxBoxes);
    currBoxes = boxes(sI,:);
    
    
    [~, goodBoxesI] = BoxNMS(currBoxes, nmsTTest);
    currBoxes = currBoxes(goodBoxesI,:);
    currScores = currScores(goodBoxesI,:);
    
    results.boxes{cI} = gather(currBoxes);
    results.scores{cI} = gather(currScores);
end