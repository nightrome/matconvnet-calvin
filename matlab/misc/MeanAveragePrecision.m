function [map, ap, conf] = MeanAveragePrecision(values, labels)
% [map ap conf] = MeanAveragePrecision(values, labels)
% *Interpolated* average precision (Pascal-style)
%
% Gets Average Precision per concept and creates a confusion matrix
%
% values:           N x M output of the classifier for each of the N concepts
%                   for M classes. A higher output means a higher rank.
% labels:           N x M double matrix of labels. 0 means undetermined
%
% map:              Mean Average Precision over all classes
% ap:               1 x M array with Average Precision per class
% conf:             M x M confusion matrix. Rows add up to one.

if islogical(labels)
    labels = double(labels);
    labels(labels == 0) = -1;
end

ap = zeros(size(labels,2), 1);
conf = zeros(size(labels,2));

for cI = 1:size(labels,2)
    gt = labels(:,cI);
    [~, si] = sort(values(:,cI), 'descend');
    tp=gt(si)>0;
    fp=gt(si)<0;
    
    % Keep all classes analogous to 'tp'. Note that this contains 'tp'
    allClasses = double(labels(si,:) > 0);
    posVector = zeros(1, size(labels,2));
    posVector(cI) = 1;
    allClasses(tp,:) = repmat(posVector, sum(tp), 1); % No loss for correctly classified images
    difficult = ~tp & ~fp;
    allClasses(difficult,:) = zeros(sum(difficult), size(labels,2)); % No loss for difficult classes
    allClassesNorm = repmat(sum(allClasses, 2), 1, size(allClasses,2)); % Normalization factor
    allClassesNorm(allClassesNorm == 0) = 1; % Avoid division by zero
    allClasses = allClasses ./ allClassesNorm; % Normalize to divide loss between classes

    allClasses = cumsum(allClasses, 1);
    fp=cumsum(fp);
    tp=cumsum(tp);
    rec=tp/sum(gt>0);
    prec=tp./(fp+tp);

    allClassesPrec = allClasses ./ repmat(fp+tp, 1, size(allClasses,2));
    [ap(cI), conf(cI,:)] = VOCapConfusion(rec, prec, allClassesPrec);
end

map = mean(ap);
    
    
function [ap, conf] = VOCapConfusion(rec,prec, falseClass)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
falseClass = cat(1, zeros(1,size(falseClass,2)), falseClass, ...
                    zeros(1,size(falseClass,2))); % Similar to mpre
for i=numel(mpre)-1:-1:1
%     mpre(i)=max(mpre(i),mpre(i+1));
    if mpre(i) < mpre(i+1)
        mpre(i) = mpre(i+1);
        falseClass(i,:) = falseClass(i+1,:);
    end
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
conf=sum(repmat(mrec(i)-mrec(i-1), 1, size(falseClass,2)) .* falseClass(i,:));

    