% Trains and tests the Fast-RCNN object detection method on PASCAL VOC 20xx TBD.
%
% Copyright by Holger Caesar, 2016

% Add folders to path
setup();

% Settings
expNameAppend = 'testRelease';
global glBaseFolder glDatasetFolder glFeaturesFolder; % Define global variables to be used in all scripts
rootFolder = calvin_root();
glBaseFolder = fullfile(rootFolder, 'data');
glDatasetFolder = fullfile(glBaseFolder, 'Datasets');
glFeaturesFolder = fullfile(glBaseFolder, 'Features');
% labelingsFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'E2S2', dataset.name, 'Run1', sprintf('%s_e2s2_run1_exp1', dataset.name), 'labelings-test-epoch-30');

global MYDATADIR;
MYDATADIR = [fullfile(glBaseFolder, 'Datasets', 'VOC2010'), '/'];

% Download dataset
downloadVOC2010();

% Download base network
downloadNetwork();

% Download Selective Search
downloadSelectiveSearch();

% Extract Selective Search regions
setupFastRcnnRegions();

% Train and test network
calvinNNDetection();

% % Show example segmentation
% fileList = dirSubfolders(labelingsFolder);
% image = imread(fullfile(labelingsFolder, fileList{1}));
% figure();
% imshow(image);