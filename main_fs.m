
clear, clc, close;

load rt-iot2022.mat; % load IoTID20.mat;

% Set 20% data as validation set
ho = 0.2;

%% Common parameter settings 
opts.N  = 20;     % number of solutions
opts.T  = 100;    % maximum number of iterations
numOfRun = 10; % number of runs

%%
opts.c1 = 0.2;
opts.c2 = 0.2;
opts.w = 0.9;

%% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);
opts.Model = HO;

%%
algorithms = ["CuckooSearch(1)","SecretaryBird(2)","HarrisHawk(3)","ParticleSwarm(4)"];
%% 
for a = 1:numel(algorithms)
     
        for r = 1:numOfRun

            disp([sprintf("\n\n %s, %s.Run",algorithms(a),num2str(r))]);
            
            tic
            switch a
                case 1
                    AlgResults = CuckooSearchAlgorithm_tstf(feat,label,opts);                                       
                case 2                    
                    AlgResults = BinarySecretaryBirdOptAlg_tstf(feat,label,opts);                                               
                case 3                    
                    AlgResults = HarrisHawksOptimization_tstf(feat,label,opts);
                case 4                    
                    AlgResults = ParticleSwarmOptimization_tstf(feat,label,opts);
            end

            res           = struct();
            res.Times     = toc/60;
            res.Algorithm = a;
            res.Run       = r;

            res.sf        = AlgResults.sf;  % Index of selected features
            res.ff        = AlgResults.ff;  % Selected features
            res.nf        = AlgResults.nf;  % Number of selected features
            res.c         = AlgResults.c;   % Convergence curve
            res.f         = AlgResults.f;   % feat
            res.l         = AlgResults.l;   % label  
            res.acc       = AlgResults.acc; % Accuracy
            
            filename = sprintf('output_files\\Result_alg%s_run%d.mat',algorithms(a),r); 
            save(filename,'res'); 
        end
end
