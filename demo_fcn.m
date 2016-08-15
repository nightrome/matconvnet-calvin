% demo_fcn()
%
% Trains and tests a Fully Convolutional Network on SIFT Flow.
%
% Copyright by Holger Caesar, 2016

% Add folders to path
startup();

% Settings
expNameAppend = 'testRelease';

% Define global variables
global glFeaturesFolder;
labelingsFolder = fullfile(glFeaturesFolder, 'CNN-Models', 'FCN', 'SiftFlow', sprintf('fcn16s-%s', expNameAppend), 'labelings-test-epoch-50');

% Download dataset
downloadSiftFlow();

% Download base network
downloadNetwork();

% Train network
fcnTrainGeneric('expNameAppend', expNameAppend);

% Test network
stats = fcnTestGeneric('expNameAppend', expNameAppend);
disp(stats);

% Show example segmentation
fileList = dirSubfolders(labelingsFolder);
image = imread(fullfile(labelingsFolder, fileList{1}));
figure();
imshow(image);
