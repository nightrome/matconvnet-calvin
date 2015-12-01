classdef ImdbDetectionFullSupervision < ImdbMatbox
    properties(SetAccess = protected, GetAccess = public)
        negOverlapRange = [0.1 0.5];
        boxesPerIm = 64;
        boxRegress = true;
    end
    methods
        function obj = ImdbDetectionFullSupervision(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm)
            obj@ImdbMatbox(imageDir, imExt, matboxDir, filenames, datasetIdx, meanIm);
        end
        
        function [batchData, numElements] = getBatch(obj, batchInds, net)
            if length(batchInds) > 1
                error('Only supports batches of 1');
            end
            
            % Load image. Make correct size. Subtract average im.
            [image, oriImSize] = obj.LoadImage(batchInds, strcmp(net.device, 'gpu'));
            
            % Sample boxes
            gStruct = obj.LoadGStruct(batchInds);
            
            if obj.flipLR
                currImT = fliplr(image);
                currBoxesT = gStruct.boxes;
                currBoxesT(:,3) = oriImSize(2) - gStruct.boxes(:,1) + 1;
                currBoxesT(:,1) = oriImSize(2) - gStruct.boxes(:,3) + 1;
                gStruct.boxes = currBoxesT;
                image = currImT;
            end
            
            if ismember(obj.datasetMode, {'train', 'val'})
                [boxes, labels, keys, overlapScores, regressionFactors] = obj.SamplePosAndNegFromGstruct(gStruct, obj.boxesPerIm);
%                 keys

                % Assign elements to cell array for use in training the network
                numElements = obj.boxesPerIm;
                if obj.boxRegress
                    batchData{10} = regressionFactors;
                    batchData{9} = 'regressionTargets';
                end
                batchData{8} = oriImSize;
                batchData{7} = 'oriImSize';
                batchData{6} = boxes;
                batchData{5} = 'boxes';
                batchData{4} = labels;
                batchData{3} = 'label';
                batchData{2} = image;
                batchData{1} = 'input';
            else
                % Test set. Get all boxes
                numElements = size(gStruct.boxes,1);
                batchData{6} = oriImSize;
                batchData{5} = 'oriImSize';
                batchData{4} = gStruct.boxes;
                batchData{3} = 'boxes';
                batchData{2} = image;
                batchData{1} = 'input';
            end
            
        end
        
        function [image, oriImSize] = LoadImage(obj, batchIdx, gpuMode)
            % image = LoadImage(obj, batchIdx)
            % Loads an image from disk, resizes it, and subtracts the mean image
            image = single(imread([obj.imageDir obj.data.(obj.datasetMode){batchIdx} obj.imExt]));
            oriImSize = double(size(image));
            image = image - imresize(obj.meanIm, [oriImSize(1) oriImSize(2)]); % Subtract mean im
            
            resizeFactor = 1000 / max(oriImSize(1:2));
            % resizeFactorMin = 600 / min(oriImSize(1:2));
            % resizeFactor = min(resizeFactorMin, resizeFactorMax);
            if gpuMode
                image = gpuArray(image);
                image = imresize(image, resizeFactor);
            else
                image = imresize(image, resizeFactor, 'bilinear', 'antialiasing', false);
            end
            
%             % Subtract mean image
%             meanIm = imresize(obj.meanIm, [size(image,1) size(image,2)], 'bilinear', 'antialiasing', false);
%             if gpuMode
%                 meanIm = gpuArray(meanIm);
%             end
%             image = image - meanIm;
        end
        
        
        function [boxes, labels, keys, overlapScores, regressionTargets] = SamplePosAndNegFromGstruct(obj, gStruct, numSamples)
            % Get positive, negative, and true GT keys
            [maxOverlap, classOverlap] = max(gStruct.overlap, [], 2);

            posKeys = find(maxOverlap >= 0.5 & gStruct.class == 0);
            negKeys = find(maxOverlap < obj.negOverlapRange(2) & maxOverlap >= obj.negOverlapRange(1) & gStruct.class == 0);
            gtKeys = find(gStruct.class > 0);

            % Get correct number of positive and negative samples
            numExtraPos = numSamples * obj.posFraction - length(gtKeys);
            numExtraPos = min(numExtraPos, length(posKeys));
            if numExtraPos > 0
                posKeys = posKeys(randperm(length(posKeys), numExtraPos));
            else
               numExtraPos = 0;
            end
            numNeg = numSamples - numExtraPos - length(gtKeys);
            numNeg = min(numNeg, length(negKeys));
            negKeys = negKeys(randperm(length(negKeys), numNeg));

            % Concatenate for final keys and labs
            keys = cat(1, gtKeys, posKeys, negKeys);
            labels = cat(1, gStruct.class(gtKeys), classOverlap(posKeys), zeros(numNeg, 1));
            labels = single(labels + 1); % Add 1 for background class
            boxes = gStruct.boxes(keys,:);

            overlapScores = cat(1, ones(length(gtKeys),1), maxOverlap(posKeys), maxOverlap(negKeys));
            
            % Calculate regression targets.
            % Jasper: I simplify Girshick by implementing regression through four
            % scalars which scale the box with respect to its center.
            if nargout == 5
                regressionTargets = zeros(size(boxes), 'like', boxes);
                regressionTargets(1:length(gtKeys),:) = 1; % Scaling factors are 1 for GT
                
                % Now get scaling factors for non-GT positive boxes
                gtBoxes = gStruct.boxes(gtKeys,:);
                posBoxes = gStruct.boxes(posKeys,:);
                for bI = 1:length(posKeys)
                    currPosBox = posBoxes(bI,:);
                    [~, gtI] = BoxBestOverlap(gtBoxes, currPosBox);
                    currGtBox = gtBoxes(gtI,:);
                    regressionTargets(bI + length(gtKeys),:) = BoxRegressionTarget(currGtBox, currPosBox);
                end
            end 
        end
        
        function SetBoxRegress(obj, doRegress)
            obj.boxRegress = doRegress;
        end
    end % End methods
end % End classdef