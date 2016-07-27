% Trains and tests a Region-based semantic segmentation with end-to-end training on SIFT Flow.
%
% Copyright by Holger Caesar, 2016

% Settings
expNameAppend = 'testRelease';

% Define global variables
global glBaseFolder glDatasetFolder glFeaturesFolder;
glBaseFolder = 'data';
glDatasetFolder = fullfile(glBaseFolder, 'Datasets');
glFeaturesFolder = fullfile(glBaseFolder, 'Features');
dataset = SiftFlowDataset();
labelingsFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'E2S2', dataset.name, sprintf('fcn16s-%s', expNameAppend), 'labelings-test-epoch-50');

% Download dataset
downloadSiftFlow();

% Download base network
downloadNetwork();

% Download Selective Search
downloadSelectiveSearch();

% Extract region proposals and labels
setupE2S2Regions('dataset', dataset);

% Train and test network
e2s2_wrapper_SiftFlow('dataset', dataset);

% % Show example segmentation
% fileList = dirSubfolders(labelingsFolder);
% image = imread(fullfile(labelingsFolder, fileList{1}));
% figure();
% imshow(image);