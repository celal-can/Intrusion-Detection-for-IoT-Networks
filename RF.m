function [ROC,METRICS]=RF(sFeat,label)
NumTrees = 100;
result = zeros(10,7);
indices = crossvalind('Kfold',label,10);

XX = [];
YY = [];
AUC = [];

for i = 1:10
    test = (indices == i);
    yvalid = label(test);
    train = ~test;
    xvalid = label(train==1);

    Mdl = fitcensemble(sFeat(train,:),xvalid,'Method','Bag','NLearn',NumTrees); %returns the trained classification ensemble model
    [pred,score] = predict(Mdl,sFeat(test,:));
    [Xforest,Yforest,Tsvm,AUCforest] = perfcurve(boolean(yvalid),score(:,boolean(Mdl.ClassNames)),'true');

    XX = [XX transpose(Xforest)];
    YY = [YY transpose(Yforest)];
    AUC = [AUC AUCforest];

    result(i,:) = Evaluate(yvalid,pred);
end
 
METRICS.ACC = mean(result(:,1));% Accuracy
METRICS.SENS = mean(result(:,2));% Sensitivity
METRICS.SPES = mean(result(:,3));% Specifivity
METRICS.PRES = mean(result(:,4));% Precision
METRICS.RECALL = mean(result(:,5));% Recalll
METRICS.FM = mean(result(:,6));% F Measure
METRICS.GMEAN = mean(result(:,7));% GMean
ROC.X = XX;
ROC.Y = YY;
ROC.AUC = AUC;
end