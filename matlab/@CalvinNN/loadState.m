function [net, stats] = loadState(fileName)
% [net, stats] = loadState(fileName)

netStruct = load(fileName, 'net', 'stats');
net = dagnn.DagNN.loadobj(netStruct.net);
stats = netStruct.stats;