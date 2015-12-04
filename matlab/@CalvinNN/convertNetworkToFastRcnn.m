function convertNetworkToFastRcnn(obj)
% convertNetworkToFastRcnn(obj)
%
% Modify network for Fast R-CNN's ROI pooling
%
% Copyright by Holger Caesar, 2015

% Replace max-pooling layer by ROI pooling
fc6Idx = obj.net.getLayerIndex('fc6');
roiPoolSize = obj.net.layers(fc6Idx).block.size(1:2);
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
obj.net.replaceLayer('pool5', 'roipool5', roiPoolBlock, {'oriImSize', 'boxes'}, {'roiPool5Mask'});

% If required, insert freeform pooling layer after roipool
if isfield(obj.nnOpts.misc, 'roiPool'),
    roiPool = obj.nnOpts.misc.roiPool;
    if isfield(roiPool, 'freeform') && roiPool.freeform.use
        % Compute activations for foreground and entire box separately
        % (by default off)
        roiPoolFreeformBlock = dagnn.RoiPoolingFreeform('combineFgBox', roiPool.freeform.combineFgBox);
        obj.net.insertLayer('roipool5', 'fc6', 'roipoolfreeform5', roiPoolFreeformBlock, 'blobMasks');
        
        % Share fully connected layer weights for foreground and entire box
        % (by default on)
        if isfield(roiPool.freeform, 'shareWeights') && ~roiPool.freeform.shareWeights
            relLayers = {'fc6', 'fc7', 'fc8'};
            for relIdx = 1 : numel(relLayers),
                relLayer = relLayers{relIdx};
                relLayerIdx = obj.net.getLayerIndex(relLayer);
                paramIndexes = obj.net.layers(relLayerIdx).paramIndexes;
                
                % Duplicate input size for all but the first fc layer
                if relIdx > 1
                    obj.net.layers(relLayerIdx).block.size(3) = obj.net.layers(relLayerIdx).block.size(3) * 2;
                    obj.net.params(paramIndexes(1)).value = cat(3, ...
                        obj.net.params(paramIndexes(1)).value, ...
                        obj.net.params(paramIndexes(1)).value);
                end
                % Duplicate output size for all but the last fc layers
                if relIdx < numel(relLayers)
                    obj.net.layers(relLayerIdx).block.size(4) = obj.net.layers(relLayerIdx).block.size(4) * 2;
                    obj.net.params(paramIndexes(1)).value = cat(4, ...
                        obj.net.params(paramIndexes(1)).value, ...
                        obj.net.params(paramIndexes(1)).value);
                    obj.net.params(paramIndexes(2)).value = cat(2, ...
                        obj.net.params(paramIndexes(2)).value, ...
                        obj.net.params(paramIndexes(2)).value);
                end
            end
        end
    end
end;