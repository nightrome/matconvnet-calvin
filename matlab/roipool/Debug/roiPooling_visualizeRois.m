function roiPooling_visualizeRois(boxes, oriImSize, convIm, rois, channelIdx, boxIdx)
% roiPooling_visualizeRois(boxes, oriImSize, convIm, rois, channelIdx, boxIdx)
%
% Visualize the input and output of the forward pass.
%
% Copyright by Holger Caesar, 2015

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
reshapedBox = (...
    (boxes(boxIdx, :) - 1) ...
    ./ ([oriImSize(2), oriImSize(1), oriImSize(2), oriImSize(1)] - 1) ...
    .* ([size(convIm, 2), size(convIm, 1), size(convIm, 2), size(convIm, 1)] - 1) ...
    );
reshapedBox = 1 + [floor(reshapedBox(1:2)), ceil(reshapedBox(3:4))];

% Get relevant "image" regions
convAll = convIm(:, :, channelIdx);
convSelection = convIm(reshapedBox(2):reshapedBox(4), reshapedBox(1):reshapedBox(3), channelIdx);
roisSelection = rois(:, :, channelIdx, boxIdx);

% Check if convolutional map is all zeros
if sum(convSelection(:)) == 0,
    clims = [0, 1];
else
    range1 = minmax(convAll(:)');
    range2 = minmax(convSelection(:)');
    range3 = minmax(roisSelection(:)');
    clims = minmax([range1, range2, range3]);
end;

% Init figure
global roiPoolFigure;
if ~isempty(roiPoolFigure) && isvalid(roiPoolFigure),   
    figure(roiPoolFigure);
else
    roiPoolFigure = figure(7);
end;

% Plot entire convolutional image
subplot(2, 3, 1);
imagesc(convAll, clims);
x1 = reshapedBox(1)-0.5;
x2 = reshapedBox(3)+0.5;
y1 = reshapedBox(2)-0.5;
y2 = reshapedBox(4)+0.5;
line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', 'Color', 'r', 'LineWidth', 2);
title('Convolutional Map (all)');

% Plot relevant section of convolutional image
subplot(2, 3, 2);
imagesc(convSelection, clims);
title('Convolutional Map (region)');

% Plot ROI
subplot(2, 3, 3);
imagesc(roisSelection, clims);
title('ROI');

% Make sure the figure is updated
drawnow();