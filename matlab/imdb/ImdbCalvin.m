classdef ImdbCalvin < handle
    %IMDBCALVIN
    % Base image database that holds information about the
    % dataset and various retrieval functions, such as getBatch(..).
    %
    % Copyright by Holger Caesar & Jasper Uijlings, 2015
    
    properties
        numClasses
        datasetMode % train, val or test

        data        % data.train data.val data.test
    end
    
    methods (Abstract)
        % This is the main method which needs to be implemented.
        % It is used by CalvinNN.train()
        batchData = getBatch(obj, batchInds, net);
    end
    
    methods
        
        function setDatasetMode(obj, datasetMode)
            % 'train', 'val', or 'test' set
            if ~ismember(datasetMode, {'train', 'val', 'test'}),
                error('Unknown datasetMode');
            end
            
            obj.datasetMode = datasetMode;
        end
        
        function allBatchIds = getAllBatchIds(obj)
            % Obtain the indices and ordering of all batches (for this epoch)
            switch obj.datasetMode
                case 'train'
                    allBatchIds = randperm(size(obj.data.train,1));
                otherwise
                    allBatchIds = 1:size(obj.data.(obj.datasetMode),1);
            end
        end
    end
end

