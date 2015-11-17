function[rois, masks, dzdxout] = roiPooling_wrapper(convIm, oriImSize, boxes, roiPoolSize, isForward, masks, dzdx)
% [rois, masks, dzdxout] = roiPooling_wrapper(convIm, oriImSize, boxes, roiPoolSize, isForward, masks, dzdx)
%
% Region of Interest pooling layer for Fast R-CNN.
% Implements both the forward and backward pass, depending on the input
% parameters. boxes are scaled from the original image size to the
% downsized convolutional image.
%
% Forward:
% [rois, mask] = roiPooling_wrapper(convIm, oriImSize, boxes, roiPoolSize)
%
% Backward:
% [~, ~, dzdx] = roiPooling_wrapper(convImSize, oriImSize, boxes, roiPoolSize, masks, dzdx)
%
% Copyright by Holger Caesar, 2015

% Debug: Visualize channels
% roiPooling_visualizeConvChannels(convIm);

if isForward,
    % Dummy init
    dzdxout = [];
    channelCount = size(convIm, 3);
    gpuMode = isa(convIm, 'gpuArray');
    
    % Move inputs from GPU if necessary
    if gpuMode,
        convIm = gather(convIm);
    end;
    
    % Perform ROI max-pooling (only works on CPU)
    [rois, masks] = roiPooling_forward(convIm, oriImSize, boxes, roiPoolSize);
    
    % Move outputs to GPU if necessary
    if gpuMode,
        rois = gpuArray(rois);
    end;
    
    % Debug: Visualize ROIs
    %     roiPooling_visualizeRois(boxes, oriImSize, convIm, rois, 1, 1);
    
    % Check size
    assert(all([size(rois, 1), size(rois, 2), size(rois, 3)] == [roiPoolSize, channelCount]));
else
    % Dummy init
    rois = [];
    boxCount = size(boxes, 1);
    convImSize = convIm;
    gpuMode = isa(dzdx, 'gpuArray');
    
    % Move inputs from GPU if necessary
    if gpuMode,
        dzdx = gather(dzdx);
    end;
    
    % Backpropagate derivatives (only works on CPU)
    dzdxout = roiPooling_backward(boxCount, convImSize, roiPoolSize, masks, dzdx);
    
    % Move outputs to GPU if necessary
    if gpuMode,
        dzdxout = gpuArray(dzdxout);
    end;
    
    % Debug: Visualize gradients
    %     roiPooling_visualizeBackward(oriImSize, boxes, masks, dzdx, dzdxout, 1, 1);
end;