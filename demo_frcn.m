% demo_frcn()
%
% Trains and tests the Fast-RCNN object detection method on PASCAL VOC 20xx TBD.
%
% Copyright by Holger Caesar, 2016

% Add folders to path
startup();

% Settings
expNameAppend = 'testRelease';
global glBaseFolder;

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