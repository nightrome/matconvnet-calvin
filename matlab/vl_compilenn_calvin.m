function vl_compilenn_calvin()

root = vl_rootnn();
mexDir = fullfile(root, 'matlab', 'mex');
mexOpts = {'-largeArrayDims', '-outdir', sprintf('"%s"', mexDir)};

mex(mexOpts{:}, fullfile(root, 'matlab', 'regiontopixel', 'regionToPixel_backward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_forward.cpp'));
mex(mexOpts{:}, fullfile(root, 'matlab', 'roipool', 'roiPooling_backward.cpp'));
