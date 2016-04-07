function state = accumulate_gradients(obj, state, net, batchSize, mmap)
% state = accumulate_gradients(obj, state, net, batchSize, mmap)
%
% Perform a Stochastic Gradient Descent update step of the network weights
% using momentum and weight decay.
%
% Copyright by Matconvnet (cnn_train_dag.m)
% Modified by Holger Caesar, 2016

for p=1:numel(net.params)
    % bring in gradients from other GPUs if any
    if ~isempty(mmap)
        numGpus = numel(mmap.Data) ;
        tmp = zeros(size(mmap.Data(labindex).(net.params(p).name)), 'single') ;
        for g = setdiff(1:numGpus, labindex)
            tmp = tmp + mmap.Data(g).(net.params(p).name) ;
        end
        net.params(p).der = net.params(p).der + tmp ;
    else
        numGpus = 1 ; %#ok<NASGU>
    end
    
    switch net.params(p).trainMethod
        
        case 'average' % mainly for batch normalization
            thisLR = net.params(p).learningRate ;
            net.params(p).value = ...
                (1 - thisLR) * net.params(p).value + ...
                (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
            
        case 'gradient'
            thisDecay = obj.nnOpts.weightDecay * net.params(p).weightDecay ;
            thisLR = state.learningRate * net.params(p).learningRate ;
            state.momentum{p} = obj.nnOpts.momentum * state.momentum{p} ...
                - thisDecay * net.params(p).value ...
                - (1 / batchSize) * net.params(p).der ;
            net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;
            
        case 'otherwise'
            error('Unknown training method ''%s'' for parameter ''%s''.', ...
                net.params(p).trainMethod, ...
                net.params(p).name) ;
    end
end