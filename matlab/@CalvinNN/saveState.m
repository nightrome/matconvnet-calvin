function saveState(fileName, net, stats) %#ok<INUSD>
% saveState(fileName, net, stats)

net = net.saveobj() ; %#ok<NASGU>
save(fileName, 'net', 'stats') ;