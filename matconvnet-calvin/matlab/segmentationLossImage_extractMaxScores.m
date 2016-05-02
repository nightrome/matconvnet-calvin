function[scoresImageSoftmax] = segmentationLossImage_extractMaxScores(obj, labelCount, sampleCount, imageCount)

% Init
scoresImageSoftmax = nan(1, 1, labelCount, sampleCount, 'like', obj.scoresMapSoftmax);
obj.mask = nan(sampleCount, 1, 'like', obj.scoresMapSoftmax); % contains the coordinates of the pixel with highest score per class

% Process each image/crop separately % very slow (!!)
if false
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
            obj.mask(sampleIdx, 1) = ind;
        end
    end
else
    % Vectorized version: No speedup :(
    for imageIdx = 1 : imageCount        
        % Find position of pixel with highest score
        s = obj.scoresMapSoftmax(:, :, :, imageIdx);
        [yVals, ys] = max(s, [], 1);
        [~, xs] = max(yVals, [], 2);
        xs = xs(:);
        ysInds = sub2ind(size(ys), ones(size(xs)), xs, (1:numel(xs))');
        ys = ys(ysInds);
        
        % Copy the scores of that pixel
        sampleStart = 1 + (imageIdx-1) * labelCount;
        sampleEnd = imageIdx * labelCount;
        a = repmat(ys, [1, labelCount]);
        b = repmat(xs, [1, labelCount]);
        c = repmat((1:labelCount)', [1, labelCount]);
        d = ones(labelCount, labelCount) * imageIdx;
        rightInds = sub2ind(size(obj.scoresMapSoftmax), a, b, c, d);
        scoresImageSoftmax(:) = obj.scoresMapSoftmax(rightInds);
        
        % Save the coordinates in a mask for the backward pass
        obj.mask(sampleStart:sampleEnd, 1) = sub2ind([size(obj.scoresMapSoftmax, 1), size(obj.scoresMapSoftmax, 2)], ys, xs);
    end
end