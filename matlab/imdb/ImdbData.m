classdef ImdbData < ImdbCalvin
    % ImdbData is a class which contains all the raw data in memory.
    % Can be used for toy examples such as the MNIST dataset
    
    properties
        labs
    end
    
    methods
        function obj = ImdbData(trainData, trainLabels) %, valData, valLabels, testData, testLabels)
%             obj.data.train = trainData;
%             obj.labs.train = trainLabels;
%             obj.data.val = valData;
%             obj.labs.val = valLabels;
%             obj.data.test = testData;
%             obj.data.val = v

        % Jasper: Just hacked for quick tests
            randIdx = randperm(size(trainData,4));
            numTrain = round(length(randIdx) * .8);
            obj.data.train = trainData(:,:,:,randIdx(1:numTrain));
            obj.labs.train = trainLabels(randIdx(1:numTrain));
            obj.data.val   = trainData(:,:,:,randIdx(numTrain+1:end));
            obj.labs.val   = trainLabels(randIdx(numTrain+1:end));
        end
        
        function batchData = getBatch(obj, batchInds, ~)
            batchData{1} = 'input';
            batchData{2} = obj.data.(obj.datasetMode)(:,:,:,batchInds);
            batchData{3} = 'label';
            batchData{4} = obj.labs.(obj.datasetMode)(batchInds);
        end
        
        function allBatchIds = getAllBatchInds(obj)
            % Obtain the indices and ordering of all batches (for this epoch)
            switch obj.datasetMode
                case 'train'
                    allBatchIds = randperm(size(obj.data.train,4));
                otherwise
                    allBatchIds = 1:size(obj.data.(obj.datasetMode),4);
            end
        end
            
    end
end

