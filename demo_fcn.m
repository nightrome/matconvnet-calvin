
% Settings
expNameAppend = 'testRelease';

% Define global variables
global glBaseFolder glDatasetFolder glFeaturesFolder; %#ok<NUSED>
%glBaseFolder = '';
glDatasetFolder = '/home/holger/Documents/Datasets';
glFeaturesFolder = '/home/holger/Documents/Features';

% Train network
fcnTrainGeneric('expNameAppend', expNameAppend);

% Test network
fcnTestGeneric('expNameAppend', expNameAppend);