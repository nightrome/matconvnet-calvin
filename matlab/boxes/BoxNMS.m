function [boxesOut, idx] = BoxNMS(boxesIn, nmsT)
% function [boxesOut idx] = BoxNMS(boxesIn, nmT)
%
% Performs non-maximum suppression: all boxes which have an overlap higher than
% maxScore with another box higher in the boxesIn list will be filtered
% out. The boxes must be already ordered
%
% boxesIn:              N x 4 array of boxes
% maxScore:             Maximum overlapping value
%
% boxesOut:             M x 4 array of boxes, M <= N
% idx:                  Logical array denoting kept boxes
%
% Jasper Uijlings - 2013


if nargin < 2
    nmsT = 0.3;
end

numBoxes = size(boxesIn,1);
isGood = true(numBoxes, 1);

[~, ~, boxSizes] = BoxSize(boxesIn);

for i=1:numBoxes-1
    if isGood(i) % Remove near duplicates
        % Get target boxes (lower than current box)
        targetI = find(isGood);
        targetI = targetI(targetI > i);
        targetBoxes = boxesIn(targetI,:);
        
        % Get size of intersection
        intersectR = (1 + min(boxesIn(i,3), targetBoxes(:,3)) - max(boxesIn(i,1), targetBoxes(:,1)));
        intersectC = (1 + min(boxesIn(i,4), targetBoxes(:,4)) - max(boxesIn(i,2), targetBoxes(:,2)));
        
        doesIntersect = intersectR > 0 & intersectC > 0;
        intersectSizes = zeros(size(intersectR));
        intersectSizes(doesIntersect) = intersectR(doesIntersect) .* intersectC(doesIntersect);
        
        % Calculate Intersection / Union. 
        scores = intersectSizes ./ (boxSizes(targetI) + boxSizes(i) - intersectSizes);
        isGoodT = scores < nmsT;
        
        isGood(targetI) = isGood(targetI) & isGoodT;
    end
end

idx = isGood;
boxesOut = boxesIn(isGood,:);