function Y = vl_nnloss_regress(X, t, dzdy, varargin)
% vl_nnloss_regress(X, t, dzdy, varargin)
%
% Performs loss incurred by the predicted values X with respect
% to the target values t.
%
% It is assumed that each vector in X is aligned w.r.t. its last dimension:
% 2-dimensional vectors are row vectors
% 3-dimensional vectors are aligned in the z-axis
% 
% Possible loss: {'L1', 'L2', and 'Smooth'}, where Smooth is L2 when |X| < 1 and L1 otherwise
% Smooth was used by Girshick to be less insensitive to outliers while having more
% sensible gradient updates close to the target
% SMOOTH NOT IMPLEMENTED YET!
%
% WEIGHTING NOT IMPLEMENTED YET!
%
% Jasper - 2015

% Determine loss 
opts.loss = 'L2';
vl_argparse(opts, varargin);

% Display warning once
warning('NotTested:regressloss', ...
    'No loss has been thoroughly tested yet');
warning('off', 'NotTested:regressloss');

assert(isequal(size(X), size(t)));

% Just compute loss
if nargin == 2 || isempty(dzdy)
    switch lower(opts.loss)
        case 'l2'
            diff = X - t;
            Y = sum(diff .* diff / 2, ndims(X)); % Sum over last dimension
        case 'l1'
            Y = sum(abs(X - t), ndims(X));
        otherwise
            error('Incorrect loss: %s', opts.loss);
    end
    Y = sum(Y); % Y = instanceWeights(:)' * t(:) ;
else
    % Get derivatives w.r.t. loss function
    switch lower(opts.loss)
        case 'l2'
            Y = permute(dzdy * (X-t), [4 3 2 1]); 
        case 'l1'
            Y = permute(dzdy * sign(X-t), [4 3 2 1]);
        otherwise
            error('Incorrect loss: %s', opts.loss);
    end
end
