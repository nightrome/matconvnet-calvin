function stats = accumulateStats(obj, stats_)
% stats = accumulateStats(obj, stats_)
%
% Goes through each GPUs struct stats_{g} and averages the values of all
% stats fields.

stats = struct();
datasetMode = obj.imdb.datasetMode;
total = 0;

for g = 1:numel(stats_)
    stats__ = stats_{g};
    num__ = stats__.(datasetMode).num;
    total = total + num__;
    
    for f = setdiff(fieldnames(stats__.(datasetMode))', 'num')
        f = char(f);
        
        if g == 1
            stats.(datasetMode).(f) = 0;
        end
        stats.(datasetMode).(f) = stats.(datasetMode).(f) + stats__.(datasetMode).(f) * num__;
        
        if g == numel(stats_)
            stats.(datasetMode).(f) = stats.(datasetMode).(f) / total;
        end
    end
end
stats.(datasetMode).num = total;