function [ROC,METRICS]=SVM(sFeat,label)

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

    Mdl = fitcsvm(sFeat(train,:),xvalid,'Standardize',true); %returns a support vector machine (SVM) classifier
    [pred,score] = predict(Mdl,sFeat(test,:));
    [Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(boolean(yvalid),score(:,boolean(Mdl.ClassNames)),'true');

    XX = [XX transpose(Xsvm)];
    YY = [YY transpose(Ysvm)];
    AUC = [AUC AUCsvm];

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