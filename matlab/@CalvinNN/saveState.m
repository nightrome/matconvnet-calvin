function saveState(fileName, net, stats) %#ok<INUSD>
% saveState(fileName, net, stats)
%
% Save network and statistics for the current epoch.
% Files are saved in v7.3 format to allow nets with > 4GB memory.

net = net.saveobj() ; %#ok<NASGU>
save(fileName, 'net', 'stats', '-v7.3') ;