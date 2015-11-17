classdef ImdbCalvin < handle
    %IMDBCALVIN
    % Base image database that holds information about the
    % dataset and various retrieval functions, such as getBatch(..).
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        numClasses
        datasetMode % train, val or test

        imagesTrain
        imagesVal
        imagesTest
    end
    
    methods (Abstract)
        batchData = getBatch(obj, batchInds, net);
    end
end

