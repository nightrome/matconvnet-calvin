function setup()
% setup()
%
% Add Matconvnet, Matconvnet-FCN and Matconvnet-Calvin to Matlab path 
% and initialize global variables used by the demos.
%
% Copyright by Holger Caesar, 2016

% Define paths
root = fileparts(mfilename('fullpath'));
matconvnetPath = fullfile(root, 'matconvnet', 'matlab');
matconvnetFcnPath = fullfile(root, 'matconvnet-fcn');
matconvnetCalvinPath = fullfile(root, 'matconvnet-calvin');

% Add matconvnet
addpath(matconvnetPath);
vl_setupnn();

% Add matconvnet-fcn
addpath(matconvnetFcnPath);

% Add matconvnet-calvin
addpath(genpath(matconvnetCalvinPath));

% Define global variables
global glBaseFolder glDatasetFolder glFeaturesFolder;
glBaseFolder = fullfile(root, 'data');
glDatasetFolder = fullfile(glBaseFolder, 'Datasets');
glFeaturesFolder = fullfile(glBaseFolder, 'Features');