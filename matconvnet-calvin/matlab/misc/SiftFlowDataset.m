classdef SiftFlowDataset < Dataset
    % SiftFlow dataset
    % Missing labels in test are 10, 12, 17
    % Copyright by Holger Caesar, 2014
    
    methods
        function[obj] = SiftFlowDataset()
            % Call superclass constructor
            obj@Dataset();
            
            global glDatasetFolder;
            
            % Dataset settings
            obj.name = 'SiftFlow';
            obj.path = fullfile(glDatasetFolder, obj.name);
            
            % Annotation settings
            annotation = Annotation('semanticLabels');
            annotation.labelFormat = 'mat-labelMap';
            annotation.imageCount = 2688;
            annotation.labelCount = 33;
            annotation.hasStuffThingLabels = true;
            obj.annotations = annotation;
            
            % Set active annotation
            obj.setActiveAnnotation('semanticLabels');
            
            % Check if dataset folder exists
            if ~exist(obj.path, 'dir'),
                error('Error: Dataset folder not found on disk!');
            end;
        end
        
        function[labelOrder, labelNames] = getLabelOrder(obj)
            % Overwrites parent function
            labelOrderNames = {'bridge', 'building', 'fence', 'crosswalk', 'sidewalk', 'road', 'field', 'grass', 'plant', 'tree', 'mountain', 'rock', 'desert', 'sand', 'moon', 'sun', 'sky', 'sea', 'river', 'awning', 'balcony', 'door', 'staircase', 'window', 'bird', 'cow', 'person', 'boat', 'car', 'bus', 'pole', 'sign', 'streetlight'};
            labelNames = obj.getLabelNames();
            labelOrder = indicesOfAInB(labelOrderNames, labelNames);
            
            if nargout > 1,
                labelNames = obj.getLabelNames();
                labelNames = labelNames(labelOrder);
            end;
        end
        
        function[imageSize] = getImageSize(~, ~)
            % [imageSize] = getImageSize(~, ~)
            %
            % Return constant image size
            
            imageSize = [256, 256];
        end
    end
end