function scores = testClassification(~, ~, net, ~)
% Get classification scores

vI = net.getVarIndex('scores');
scoresStruct = net.vars(vI);
scores = permute(scoresStruct.value, [4 3 2 1]);
