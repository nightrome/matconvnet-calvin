classdef LossUoi < dagnn.Loss
    % LossUoi < dagnn.Loss
    %
    % A Union over Intersection (UOI) loss as suggested by
    % [Combining the Best of Graphical Models and ConvNets for Semantic
    % Segmentation, Cogswell et al., arxiv 2014].
    %
    % Minimizing UOI should correspond to maximizing IOU, assuming that for
    % each class there are more than 0 true positives.
    % Requires previous softmax normalization of the scores.
    % (see dagnn.SoftMax)
    %
    % inputs: scores, labels, [instanceWeights]
    % outputs: loss
    %
    % The forward pass will output very high values if there are no true
    % positives for a class. This is irrelevant for the backward pass and
    % therefore does not affect training.
    %
    % Note: If you use instanceWeights to change the total weight of this
    % batch, then you shouldn't use the default extractStatsFn anymore, as
    % its average-loss depends on the number of boxes in the batch.
    %
    % Copyright by Holger Caesar, 2015
    
    properties (Transient)
        numSubBatches = 0;
        minScore = 1e-8; % epsilon value to avoid having 0 true positives
    end
    
    methods
        function outputs = forward(obj, inputs, params) %#ok<INUSD>
            % outputs = forward(obj, inputs, params)
            %
            % Computes expected UOI (soft version of UOI)
            
            % Get inputs
            assert(numel(inputs) == 3);
            scoresPred = inputs{1};
            labels = inputs{2};
            instanceWeights = inputs{3};
            
            labelCount = size(scoresPred, 3);
            pixelCount = size(scoresPred, 4);
            
            % Check or initialize instance weights
            if ~isempty(instanceWeights)
                assert(numel(instanceWeights) == size(scoresPred, 4));
                assert(numel(instanceWeights) == size(labels, ndims(labels)));
            else
                instanceWeights = ones(1, 1, 1, pixelCount);
            end
            
            % Reshape for convenience
            scoresPred = reshape(scoresPred, labelCount, pixelCount);
            labels = reshape(labels, 1, pixelCount);
            instanceWeights = reshape(instanceWeights, 1, pixelCount);
            
            % Make sure that the predicted scores are all > 0
            scoresPred(scoresPred == 0) = obj.minScore;
            
            % Init
            uois = nan(labelCount, 1);
            
            for labelIdx = 1 : labelCount
                % Skip non-GT labels
                if ~ismember(labelIdx, labels)
                    continue;
                end
                
                % Get the predicted and true probabilities of the
                % current pixel and class
                preds = scoresPred(labelIdx, :);
                trues = double(labels == labelIdx);
                intersections = preds .* trues;
                unions = preds + trues - intersections;
                
                intersection = sum(intersections .* instanceWeights);
                union = sum(unions .* instanceWeights);
                assert(intersection ~= 0);
                uois(labelIdx) = union / intersection;
            end
            uoi = nanmean(uois);
            outputs{1} = uoi;
            
            % Update statistics
            n = obj.numAveraged;
            m = n + size(inputs{1}, 4);
            obj.average = (n * obj.average + gather(outputs{1})) / m;
            obj.numAveraged = m;
            obj.numSubBatches = obj.numSubBatches + 1;
            assert(~isnan(obj.average));
        end
        
        function[derInputs, derParams] = backward(obj, inputs, params, derOutputs) %#ok<INUSL>
            
            % Get inputs
            assert(numel(derOutputs) == 1);
            scoresPred = inputs{1};
            labels = inputs{2};
            instanceWeights = inputs{3};
            dzdy = derOutputs{1};
            
            % Get dimensions and reshape for convenience
            labelCount = size(scoresPred, 3);
            pixelCount = size(scoresPred, 4);
            scoresPred = reshape(scoresPred, labelCount, pixelCount);
            labels = reshape(labels, 1, pixelCount);
            instanceWeights = reshape(instanceWeights, 1, pixelCount);
            
            % Init
            dzdx = zeros(labelCount, pixelCount);
            
            % Make sure that the predicted scores are all > 0
            scoresPred(scoresPred == 0) = obj.minScore;
            
            for labelIdx = 1 : labelCount,
                % Skip non-GT labels
                if ~ismember(labelIdx, labels);
                    continue;
                end
                
                % Get the predicted and true probabilities of the
                % current pixel and class
                preds = scoresPred(labelIdx, :);
                trues = double(labels == labelIdx);
                
                intersection = sum(preds .* trues); % expected intersection
                simplified = trues .* sum(preds + trues); % a heavily simplified term that is not intuitive
                
                dzdx(labelIdx, :) = (intersection - simplified) ./ (intersection .^ 2);
            end
            
            % Weight pixels if specified
            dzdy = dzdy * instanceWeights;
            dzdx = bsxfun(@times, dzdy, dzdx);
            
            % Reshape to original format
            dzdx = reshape(dzdx, 1, 1, labelCount, pixelCount);
            
            derInputs{1} = dzdx;
            derInputs{2} = [];
            derInputs{3} = [];
            derParams = {};
        end
        
        function obj = LossUoi(varargin)
            obj = obj@dagnn.Loss(varargin{:});
        end
        
        function reset(obj)
            reset@dagnn.Loss(obj);
            obj.numSubBatches = 0;
        end
        
        function forwardAdvanced(obj, layer)
            %FORWARDADVANCED  Advanced driver for forward computation
            %  FORWARDADVANCED(OBJ, LAYER) is the advanced interface to compute
            %  the forward step of the layer.
            %
            %  The advanced interface can be changed in order to extend DagNN
            %  non-trivially, or to optimise certain blocks.
            %
            % Jasper: Overrides standard forward pass to avoid giving up when any of
            % the inputs is empty.
            
            in = layer.inputIndexes;
            out = layer.outputIndexes;
            par = layer.paramIndexes;
            net = obj.net;
            
            inputs = {net.vars(in).value};
            
            % give up if any of the inputs is empty (this allows to run
            % subnetworks by specifying only some of the variables as input --
            % however it is somewhat dangerous as inputs could be legitimaly
            % empty)
            % Jasper: Indeed. Removed this option to enable not using instanceWeights
            %              if any(cellfun(@isempty, inputs)), return; end
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious && ~net.computingDerivative && net.conserveMemory
                        net.vars(v).value = [];
                    end
                end
            end
            
            %[net.vars(out).value] = deal([]);
            
            % call the simplified interface
            outputs = obj.forward(inputs, {net.params(par).value});
            [net.vars(out).value] = deal(outputs{:});
        end
    end
end