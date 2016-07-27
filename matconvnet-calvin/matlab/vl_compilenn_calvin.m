function vl_compilenn_calvin()
% vl_compilenn_calvin()
%
% Compile the C code provided in Matconvnet-calvin.
% Matconvnet code needs to be compiled separately.
%
% Copyright by Holger Caesar, 2016

root = fileparts(fileparts(mfilename('fullpath')));
mexDir = fullfile(root, 'matlab', 'mex');
mexOpts = {'-largeArrayDims', '-outdir', sprintf('"%s"', mexDir)};

% E2S2-related
mex(mexOpts{:}, fullfile(root, 'matlab', 'labelpresence', 'labelPresence_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'regiontopixel', 'regionToPixel_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_forward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_backward.cpp'));

% Misc
mex(mexOpts{:}, fullfile(root, 'matlab', 'misc', 'computeBlobOverlapAnyPair.cpp'));