function[scoresImageSoftmax, mask] = segmentationLossImage_extractMaxScores(obj, labelCount, sampleCount, imageCount)
% [scoresImageSoftmax, mask] = segmentationLossImage_extractMaxScores(obj, labelCount, sampleCount, imageCount)
%
% TODO: Mex this file

% Init
scoresImageSoftmax = nan(1, 1, labelCount, sampleCount, 'like', obj.scoresMapSoftmax);
mask = nan(sampleCount, 1, 'like', obj.scoresMapSoftmax); % contains the coordinates of the pixel with highest score per class

% Process each image/crop separately % very slow (!!)
for imageIdx = 1 : imageCount
    offset = (imageIdx-1) * labelCount;
    
    for labelIdx = 1 : labelCount
        sampleIdx = offset + labelIdx;
        
        if obj.useScoreDiffs
            s = obj.scoresMapSoftmax(:, :, labelIdx, imageIdx) - max(obj.scoresMapSoftmax(:, :, setdiff(1:labelCount, labelIdx), imageIdx), [], 3);
        else
            s = obj.scoresMapSoftmax(:, :, labelIdx, imageIdx);
        end
        [~, ind] = max(s(:)); % always take first pix with max score
        [y, x] = ind2sub(size(s), ind);
        scoresImageSoftmax(1, 1, :, sampleIdx) = obj.scoresMapSoftmax(y, x, :, imageIdx);
        mask(sampleIdx, 1) = ind;
    end
end