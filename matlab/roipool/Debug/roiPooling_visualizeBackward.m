function roiPooling_visualizeBackward(oriImSize, boxes, masks, dzdx, dzdxout, channelIdx, boxIdx)
% roiPooling_visualizeBackward(oriImSize, boxes, masks, dzdx, dzdxout, channelIdx, boxIdx)
%
% Visualize the input and output of the backward pass.
%
% Copyright by Holger Caesar, 2015

% Settings
doChecks = false;

% Default arguments
if ~exist('channelIdx', 'var'),
    channelIdx = 1;
end;
if ~exist('boxIdx', 'var'),
    boxIdx = 1;
end;

% Check inputs
assert(boxIdx <= size(boxes, 1));

% Reshape box to convIm
convImSize = [size(dzdxout, 1), size(dzdxout, 2), size(dzdxout, 3)];
reshapedBox = (...
    (boxes(boxIdx, :) - 1) ...
    ./ ([oriImSize(2), oriImSize(1), oriImSize(2), oriImSize(1)] - 1) ...
    .* ([convImSize(2), convImSize(1), convImSize(2), convImSize(1)] - 1) ...
    );
reshapedBox = 1 + [floor(reshapedBox(1:2)), ceil(reshapedBox(3:4))];

% Get relevant "image" regions
dzdxoutAll = dzdxout(:, :, channelIdx, boxIdx);
dzdxoutSelection = dzdxout(reshapedBox(2):reshapedBox(4), reshapedBox(1):reshapedBox(3), channelIdx);
dzdxSelection = dzdx(:, :, channelIdx, boxIdx);

% Take apart the masks (C-indexing!)
curMasks = masks(:, :, channelIdx, boxIdx);
curMasksY = mod(curMasks, convImSize(1));
curMasksX = (curMasks - curMasksY) / convImSize(1);
% Convert from C-indexing to Matlab
curMasksY = curMasksY + 1;
curMasksX = curMasksX + 1;

% Very expensive check to see if everything worked
% (only works if there is just a single boxes)
if doChecks,
    for imIdxY = 1 : convImSize(1),
        for imIdxX = 1 : convImSize(2),
            gradientSum = sum(sum(dzdxSelection .* (curMasksY == imIdxY & curMasksX == imIdxX)));
            condition = abs(gradientSum - dzdxoutAll(imIdxY, imIdxX)) < 1e-8;
            assert(gather(condition));
        end;
    end;
end;

% Take the absolute values of all gradients (for visualization)
dzdxoutAll = abs(dzdxoutAll);
dzdxoutSelection = abs(dzdxoutSelection);
dzdxSelection = abs(dzdxSelection);

% Check if convolutional map is all zeros
if sum(dzdxoutSelection(:)) == 0,
    clims = [0, 1];
else
    range1 = minmax(dzdxoutAll(:)');
    range2 = minmax(dzdxoutSelection(:)');
    range3 = minmax(dzdxSelection(:)');
    clims = minmax([range1, range2, range3]);
end;

% Init figure
global roiPoolFigure;
if ~isempty(roiPoolFigure) && isvalid(roiPoolFigure),   
    figure(roiPoolFigure);
else
    roiPoolFigure = figure(7);
end;

% Plot entire left-hand side gradients
subplot(2, 3, 4);
imagesc(dzdxoutAll, clims);
x1 = reshapedBox(1)-0.5;
x2 = reshapedBox(3)+0.5;
y1 = reshapedBox(2)-0.5;
y2 = reshapedBox(4)+0.5;
line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'color', 'r', 'LineWidth', 2);
title('Left-hand side gradients (all)');

% Plot relevant section of left-hand side gradients
subplot(2, 3, 5);
imagesc(dzdxoutSelection, clims);
title('Left-hand side gradients (region)');

% Plot right-hand side gradients
subplot(2, 3, 6);
imagesc(dzdxSelection, clims);
title('Right-hand side gradients');

% Plot mask coordinates
poolSizeY = size(masks, 1);
poolSizeX = size(masks, 2);
for poolIdxX = 1 : poolSizeX,
    for poolIdxY = 1 : poolSizeY,
        if ~isnan(curMasksX(poolIdxY, poolIdxX)),
            str = sprintf('[%d, %d]', curMasksY(poolIdxY, poolIdxX), curMasksX(poolIdxY, poolIdxX));
            text(poolIdxX, poolIdxY, str, 'HorizontalAlignment', 'center');
        end;
    end;
end;

% Make sure the figure is updated
drawnow();