function write_gradients(mmap, net)
% write_gradients(mmap, net)

for i=1:numel(net.params)
  mmap.Data(labindex).(net.params(i).name) = gather(net.params(i).der) ;
end