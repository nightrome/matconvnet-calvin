function mmap = map_gradients(obj, net)
% mmap = map_gradients(obj, net)

% Get inputs
fname = obj.nnOpts.memoryMapFile;
numGpus = numel(obj.nnOpts.gpus);

format = {} ;
for i=1:numel(net.params)
  format(end+1,1:3) = {'single', size(net.params(i).value), net.params(i).name} ;
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;