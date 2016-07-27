function downloadNetwork()
% downloadNetwork()
%
% Downloads the VGG-16 network model.
%
% Copyright by Holger Caesar, 2016

% Settings
modelName = 'imagenet-vgg-verydeep-16';
url = sprintf('http://www.vlfeat.org/matconvnet/models/%s.mat', modelName);
rootFolder = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
modelFolder = fullfile(rootFolder, 'data', 'Features', 'CNN-Models', 'matconvnet');
modelPath = fullfile(modelFolder, [modelName, '.mat']);

% Download network
if ~exist(modelPath, 'file')
    % Create folder
    if ~exist(modelFolder, 'dir')
        mkdir(modelFolder);
    end
    
    % Download model
    if ~exist(modelPath, 'dir')
        fprintf('Downloading model...\n');
        urlwrite(url, modelPath);
    end
end