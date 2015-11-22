function vl_setupnn_calvinn()
% Calvin way of setting up matconvnet

% Call standard setup
vl_setupnn();

root = vl_rootnn();

% New includes necessary for matconvnet-calvin
addpath(fullfile(root, 'matlab', 'imdb'));
addpath(fullfile(root, 'matlab', 'regiontopixel'));
addpath(fullfile(root, 'matlab', 'roipool'));
addpath(fullfile(root, 'matlab', 'boxes'));