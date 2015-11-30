function convertNetworkToFastRcnn(obj)
% convertNetworkToFastRcnn(obj)

% Modify network for Fast R-CNN's ROI pooling

% Replace max-pooling layer by ROI pooling
fc6Idx = obj.net.getLayerIndex('fc6');
roiPoolSize = obj.net.layers(fc6Idx).block.size(1:2);
roiPoolBlock = dagnn.RoiPooling('poolSize', roiPoolSize);
obj.net.replaceLayer('pool5', 'roipool5', roiPoolBlock, {'oriImSize', 'boxes'}, {'roiPool5Mask'});

% If required, insert freeform pooling layer after roipool
if isfield(obj.nnOpts.misc, 'roiPool'),
    roiPool = obj.nnOpts.misc.roiPool;
    if isfield(roiPool, 'freeform') && roiPool.freeform.use
        roiPoolFreeformBlock = dagnn.RoiPoolingFreeform('combineFgBox', roiPool.freeform.combineFgBox);
        obj.net.insertLayer('roipool5', 'fc6', 'roipoolfreeform5', roiPoolFreeformBlock, 'blobMasks');
    end
end;