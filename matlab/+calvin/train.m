function[] = train(net)

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('resuming by loading epoch %d\n', start) ;
  [net, stats] = loadState(modelPath(start)) ;
end

for epoch=start+1:opts.numEpochs

  % train one epoch
  state.epoch = epoch ;
  state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  state.val = opts.val ;
  state.imdb = imdb ;

  if numGpus <= 1
    stats.train(epoch) = process_epoch(net, state, opts, 'train') ;
    stats.val(epoch) = process_epoch(net, state, opts, 'val') ;
  else
    savedNet = net.saveobj() ;
    spmd
      net_ = dagnn.DagNN.loadobj(savedNet) ;
      stats_.train = process_epoch(net_, state, opts, 'train') ;
      stats_.val = process_epoch(net_, state, opts, 'val') ;
      if labindex == 1, savedNet_ = net_.saveobj() ; end
    end
    net = dagnn.DagNN.loadobj(savedNet_{1}) ;
    stats__ = accumulateStats(stats_) ;
    stats.train(epoch) = stats__.train ;
    stats.val(epoch) = stats__.val ;
  end

  % save
  if ~evaluateMode
    saveState(modelPath(epoch), net, stats) ;
  end

  figure(1) ; clf ;
  values = [] ;
  leg = {} ;
  for s = {'train', 'val'}
    s = char(s) ;
    for f = setdiff(fieldnames(stats.train)', {'num', 'time'})
      f = char(f) ;
      leg{end+1} = sprintf('%s (%s)', f, s) ;
      tmp = [stats.(s).(f)] ;
      values(end+1,:) = tmp(1,:)' ;
    end
  end
  subplot(1,2,1) ; plot(1:epoch, values') ;
  legend(leg{:}) ; xlabel('epoch') ; ylabel('metric') ;
  subplot(1,2,2) ; semilogy(1:epoch, values') ;
  legend(leg{:}) ; xlabel('epoch') ; ylabel('metric') ;
  grid on ;
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end