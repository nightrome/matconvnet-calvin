function vl_compilenn_calvin()
% vl_compilenn_calvin()
%
% Compile the C code provided in Matconvnet-calvin.
% You also need to execute vl_compilenn('EnableGpu', true).
%
% Copyright by Holger Caesar, 2016

root = vl_rootnn();
mexDir = fullfile(root, 'matlab', 'mex');
mexOpts = {'-largeArrayDims', '-outdir', sprintf('"%s"', mexDir)};

mex(mexOpts{:}, fullfile(root, 'matlab', 'labelpresence', 'labelPresence_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'labelpresence', 'segmentationLabelPresence_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'regiontopixel', 'regionToPixel_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_forward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_backward.cpp'));
