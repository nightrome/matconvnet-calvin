function[scoresSP, labelsSP, weightsSP, mapSP] = regionToPixel_forward(scoresAll, regionToPixelAux, inverseLabelFreqs, normalizeImageMass, replicateUnpureSPs)
% [scoresSP, labelsSP, weightsSP, mapSP] = regionToPixel_forward(scoresAll, regionToPixelAux, inverseLabelFreqs, normalizeImageMass, replicateUnpureSPs)
%
% Go from a region level to a pixel level.
% (to be able to compute a loss there)
%
% Note: The presence/absence of regionToPixelAux.spLabelHistos indicates
% whether we are in train/val or test mode.
%
% Copyright by Holger Caesar, 2015

% Move to CPU
gpuMode = isa(scoresAll, 'gpuArray');
if gpuMode,
    scoresAll = gather(scoresAll);
end;

% Check inputs
assert(~any(isnan(scoresAll(:)) | isinf(scoresAll(:))));

% Reshape scores
scoresAll = reshape(scoresAll, [size(scoresAll, 3), size(scoresAll, 4)]);

% Get additional batch info
overlapListAll = regionToPixelAux.overlapListAll;

% Init
labelCount = size(scoresAll, 1);
spCount = size(overlapListAll, 2);
scoresSP = nan(labelCount, spCount, 'single'); % Note that zeros will be counted anyways!
mapSP = nan(labelCount, spCount);

% Compute maximum scores and map/mask for the backward pass
for spIdx = 1 : spCount,
    ancestors = find(overlapListAll(:, spIdx));
    if ~isempty(ancestors),
        % For each label, compute the ancestor with the highest score
        [scoresSP(:, spIdx), curInds] = max(scoresAll(:, ancestors), [], 2);
        curBoxInds = ancestors(curInds);
        mapSP(:, spIdx) = curBoxInds;
    end;
end;

% Compute sample target labels and weights
splitWeightUnpureSPs = isfield(regionToPixelAux, 'spLabelHistos');
if ~splitWeightUnpureSPs,
    % Set dummy outputs
    labelsSP = [];
    weightsSP = [];
else
    % Get input fields
    labelPixelFreqs = regionToPixelAux.labelPixelFreqs;
    spLabelHistos   = regionToPixelAux.spLabelHistos;
    
    % Check inputs
    assert(all(size(labelPixelFreqs) == [labelCount, 1]));
    assert(all(size(spLabelHistos) == [spCount, labelCount]));
    
    % If an SP has no label, we need to remove it from scores, map, target and
    % pixelSizes (only in train/val)
    nonEmptySPs = ~any(isnan(scoresSP))';
    scoresSP = scoresSP(:, nonEmptySPs);
    mapSP = mapSP(:, nonEmptySPs);
    spLabelHistos = spLabelHistos(nonEmptySPs, :);
    spCount = sum(nonEmptySPs);
    assert(spCount >= 1);
    
    % Replicate regions with multiple labels
    % (change: scoresSP, labelsTargetSP, mapSP, pixelSizesSP)
    if replicateUnpureSPs,
        scoresSPRepl = cell(spCount, 1);
        labelsTargetSPRepl = cell(spCount, 1);
        mapSPRepl = cell(spCount, 1);
        pixelSizesSPRepl = cell(spCount, 1);
        for spIdx = 1 : spCount,
            replInds = find(spLabelHistos(spIdx, :))';
            replCount = numel(replInds);
            scoresSPRepl{spIdx} = repmat(scoresSP(:, spIdx)', [replCount, 1]);
            mapSPRepl{spIdx} = repmat(mapSP(:, spIdx)', [replCount, 1]);
            labelsTargetSPRepl{spIdx} = replInds;
            pixelSizesSPRepl{spIdx} = spLabelHistos(spIdx, replInds)';
        end;
        scoresSP = cell2mat(scoresSPRepl)';
        mapSP = cell2mat(mapSPRepl)';
        labelsSP = cell2mat(labelsTargetSPRepl);
        pixelSizesSP = cell2mat(pixelSizesSPRepl);
    else
        [~, labelsSP] = max(spLabelHistos, [], 2);
        pixelSizesSP = sum(spLabelHistos, 2);
    end;
    
    % Renormalize label weights to have on average a weight == 1 (!)
    % Note: division by the number of images on which these frequencies are
    % computed is now outsourced to getBatch.
    if inverseLabelFreqs,
        weightsSP = pixelSizesSP ./ (labelPixelFreqs(labelsSP) * labelCount);
    else
        weightsSP = pixelSizesSP  / sum(labelPixelFreqs);
    end;
    
    % Renormalize to (average of) 1
    if normalizeImageMass,
        weightsSP = weightsSP ./ sum(weightsSP);
    end;
    
    % Reshape and append label weights
    labelsSP  = reshape(labelsSP,  1, 1, 1, []);
    weightsSP = reshape(weightsSP, 1, 1, 1, []);
    
    % Final checks (only in train, in test NANs are fine)
    assert(~any(isnan(scoresSP(:)) | isinf(scoresSP(:))));
end;

% Reshape the scores
scoresSP = reshape(scoresSP, [1, 1, size(scoresSP)]);

% Convert outputs back to GPU if necessary
if gpuMode,
    scoresSP = gpuArray(scoresSP);
end;