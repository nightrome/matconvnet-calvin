classdef ImdbCalvin < handle
    %IMDBCALVIN
    % Base image database that holds information about the
    % dataset and various retrieval functions, such as getBatch(..).
    %
    % Copyright by Holger Caesar & Jasper Uijlings, 2015
    
    properties
        numClasses
        datasetMode % train, val or test

        imagesTrain
        imagesVal
        imagesTest
    end
    
    methods (Abstract)
        % This is the main method which needs to be implemented.
        % It is used by CalvinNN.train()
        batchData = getBatch(obj, batchInds, net);
        
        % Necessary 
        numBatches = getNumBatches(obj, datasetMode);
    end
    
    methods
        
        function setDatasetMode(obj, datasetMode)
            % 'train', 'val', or 'test' set
            if ~ismember(datasetMode, {'train', 'val', 'test'}),
                error('Unknown datasetMode');
            end
            
            obj.datasetMode = datasetMode;
        end
    end
end

