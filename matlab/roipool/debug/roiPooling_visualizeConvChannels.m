function roiPooling_visualizeConvChannels(convIm)
% roiPooling_visualizeConvChannels(convIm)
%
% Visualize the channels of the convolutional map.
%
% Copyright by Holger Caesar, 2015

% Determine number of channels to show
nonEmptyChannels = find(squeeze(sum(sum(convIm, 1), 2)) ~= 0);
nonEmptyChannelCount = min(numel(nonEmptyChannels), 25);

% Create figure
figure(6);

% Show channels
subPlotLength = ceil(sqrt(nonEmptyChannelCount));
for i = 1 : nonEmptyChannelCount,
    subplot(subPlotLength, subPlotLength, i);
    imagesc(convIm(:, :, nonEmptyChannels(i)));
end;

% Make sure the figure is updated
drawnow();