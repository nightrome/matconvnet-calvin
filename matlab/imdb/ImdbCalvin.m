classdef ImdbCalvin
    %IMDBCALVIN
    % Base image database that holds information about the
    % dataset and various retrieval functions, such as getBatch(..).
    %
    % Copyright by Holger Caesar, 2015
    
    properties
        labelCount;
        train;
        val;
        test;
    end
    
    methods (Abstract)
        batchData = getBatch(obj, batchIdx);
    end
end

