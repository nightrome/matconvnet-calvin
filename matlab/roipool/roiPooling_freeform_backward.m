function[dzdxout] = roiPooling_freeform_backward(roiPool, dzdx)
% [dzdxout] = roiPooling_freeform_backward(roiPool, dzdx)
%
% Freeform pooling backward pass.
%
% If specified, this function sums the gradients for the foreground
% mask and the entire box. Otherwise it just passed through the gradients.
%
% Copyright by Holger Caesar, 2015

% Get inputs
roiPoolFreeform = roiPool.roiPoolFreeform;
roiPoolFreeformUseFgAll = roiPool.roiPoolFreeformUseFgAll;

if roiPoolFreeform && roiPoolFreeformUseFgAll,
    % Sum the gradients from fg and entire box
    half = size(dzdx, 3) / 2;
    dzdxFg  = dzdx(:, :, 1:half, :);
    dzdxBox = dzdx(:, :, half+1:end, :);
    dzdxout = dzdxFg + dzdxBox;
else
    % Just pass through the gradients
    dzdxout = dzdx;
end;