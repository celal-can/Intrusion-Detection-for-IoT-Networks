function [ROC,METRICS]=KNN(sFeat,label)
k = 5;
result = zeros(10,7); % 10 fold ve 7 pereformans metriği için
indices = crossvalind('Kfold',label,10); % 10 fold için böl

XX = [];
YY = [];
AUC = [];

for i = 1:10
    test = (indices == i);
    yvalid = label(test);
    train = ~test;
    xvalid = label(train==1);

    Mdl = fitcknn(sFeat(train,:),xvalid,'NumNeighbors',k); % returns a k-nearest neighbor classification model
    [pred,score] = predict(Mdl,sFeat(test,:));
    [Xknn,Yknn,Tknn,AUCknn] = perfcurve(boolean(yvalid),score(:,boolean(Mdl.ClassNames)),'true');%ROC curve

    XX = [XX transpose(Xknn)];
    YY = [YY transpose(Yknn)];
    AUC = [AUC AUCknn];

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