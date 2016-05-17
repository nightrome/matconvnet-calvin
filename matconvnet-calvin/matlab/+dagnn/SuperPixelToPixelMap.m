classdef SuperPixelToPixelMap < dagnn.Layer
    % Convert pixel label scores to presence/absence scores for each class per batch.
    % (to be able to compute an image-level loss there)
    %
    % inputs are: scoresSP, blobsSP
    % outputs are: scoresImage
    %
    % Copyright by Holger Caesar, 2015
    
    properties (Transient)
        mask
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            
            % Get inputs
            assert(numel(inputs) == 2);
            scoresSP = inputs{1};
            blobsSP = inputs{2};
            labelCount = size(scoresSP, 3);
            spCount = size(scoresSP, 4);
            
            % Move to CPU
            gpuMode = isa(scoresSP, 'gpuArray');
            if gpuMode
                scoresSP = gather(scoresSP);
            end
            
            % Init
            imageSize = max(cell2mat({blobsSP.rect}')); %TODO: get the proper image size
            imageSize = imageSize(3:4);
            scoresMap = nan(imageSize(1), imageSize(2), labelCount, 1, 'like', scoresSP);
            obj.mask = cell(spCount, labelCount);
            
            for spIdx = 1 : spCount
                blob = blobsSP(spIdx);
                
                % Get all pix. coords for the mask
                [blobSubY, blobSubX] = blobToImageSubs(blob);
                
                % Copy scores to all pixels in that superpixel
                for labelIdx = 1 : labelCount
                    curInds = blobSubY + imageSize(1) * (blobSubX-1) + imageSize(1) * imageSize(2) * (labelIdx-1);
                    obj.mask{spIdx, labelIdx} = curInds;
                    scoresMap(curInds) = scoresSP(:, :, labelIdx, spIdx);
                end
            end
            
            % Convert outputs back to GPU if necessary
            if gpuMode
                scoresMap = gpuArray(scoresMap);
            end
            
            % Store outputs
            outputs = cell(1, 1);
            outputs{1} = scoresMap;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            scoresSP = inputs{1};
            labelCount = size(scoresSP, 3);
            spCount = size(scoresSP, 4);
            dzdy = derOutputs{1};
            
            % Move to CPU
            gpuMode = isa(dzdy, 'gpuArray');
            if gpuMode
                dzdy = gather(dzdy);
            end
            
            % Init
            dzdx = zeros(size(scoresSP), 'like', scoresSP);
            
            % Map pixel gradients to superpixels
            for spIdx = 1 : spCount                
                % Copy scores to all pixels in that superpixel
                for labelIdx = 1 : labelCount
                    % If any of the gradients is not 0, we copy it to the
                    % whole superpixel (otherwise it remains 0)
                    curInds = obj.mask{spIdx, labelIdx};
                    temp = dzdy(curInds);
                    temp = temp(temp ~= 0);
                    if ~isempty(temp)
                       dzdx(1, 1, labelIdx, spIdx) = temp;
                    end
                end
            end
            
            % Convert outputs back to GPU if necessary
            gpuMode = isa(scoresSP, 'gpuArray');
            if gpuMode
                dzdx = gpuArray(dzdx);
            end
            
            % Store gradients
            derInputs{1} = dzdx;
            derInputs{2} = [];
            derParams = {};
        end
        
        function obj = SuperPixelToPixelMap(varargin)
            obj.load(varargin);
        end
    end
end