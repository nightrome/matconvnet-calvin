function[roisBlob, masksBlob] = roiPooling_freeform_blob(blobMask, roisBlob, masksBlob)
% [roisBlob, masksBlob] = roiPooling_freeform_blob(blobMask, roisBlob, masksBlob)
%
% Apply a freeform mask to the outputs of Region Of Interest pooling.
%
% Copyrights by Holger Caesar, 2015

blobMaskNan = double(~blobMask);
blobMaskNan(blobMaskNan(:) == 0) = nan;

roisBlob  = bsxfun(@times, roisBlob, blobMask);
masksBlob = bsxfun(@times, masksBlob, blobMaskNan);