load rt-iot2022_orj.mat
load Results4alg.mat % Concatenated Result files

algorithms = ["CuckooSearch","SecretaryBird","HarrisHawk","ParticleSwarm"];
methods = ["KNN","SVM","RF"];

for ifSelFeat = 2 % Features selected? 1: not selected, 2: selected

    for a = 1:numel(algorithms) % loop of algorithms    
        if (ifSelFeat == 1)            
            sFeat = feat;
        elseif (ifSelFeat ==2)            
            algdata = res([res.Algorithm]==a);
            if (a==1)
                sFeat = algdata(2).ff;% select manually the best accuracy
            elseif(a==2)
                sFeat = algdata(5).ff;% select manually the best accuracy
            elseif(a==3)
                sFeat = algdata(2).ff;% select manually the best accuracy
            elseif(a==4)
                sFeat = algdata(8).ff;% select manually the best accuracy
            end
        end

        for m = 1:numel(methods) % loop of methods

            switch m
                case 1
                    [ROC,METRICS]=KNN(sFeat,label);
                case 2
                    [ROC,METRICS]=SVM(sFeat,label);
                case 3
                    [ROC,METRICS]=RF(sFeat,label);
            end

            evalRes = struct();
            
            evalRes.a = a;
            evalRes.m = m;
            evalRes.ifSelFeat = ifSelFeat;
            evalRes.ff = sFeat;
            evalRes.l = label;
            evalRes.ACC = METRICS.ACC;
            evalRes.SENS = METRICS.SENS;
            evalRes.SPES = METRICS.SPES;
            evalRes.PRES = METRICS.PRES;
            evalRes.RECALL = METRICS.RECALL;
            evalRes.FM = METRICS.FM;
            evalRes.GMEAN = METRICS.GMEAN;
            evalRes.ROCX = ROC.X;
            evalRes.ROCY = ROC.Y;
            evalRes.AUC = ROC.AUC;
            filename = sprintf('output_files_eval\\Result_alg%s_method%s_SelFeat%d.mat',algorithms(a),methods(m),ifSelFeat);
            save(filename,'evalRes');

        end % m end

    end % a end

end % s end
