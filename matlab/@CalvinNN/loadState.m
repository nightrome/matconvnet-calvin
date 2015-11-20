function loadState(obj, fileName)
% loadState(obj, fileName)

netStruct = load(fileName, 'net', 'stats');
obj.net = dagnn.DagNN.loadobj(netStruct.net);
obj.stats = netStruct.stats;