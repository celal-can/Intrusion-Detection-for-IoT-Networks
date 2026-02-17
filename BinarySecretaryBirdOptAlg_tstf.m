
% This algorithm is adapted from Fu, Y., Liu, D., Chen, J. et al. 
% Secretary bird optimization algorithm: a new metaheuristic for solving global optimization problems. Artif Intell Rev 57, 123 (2024). 
% https://doi.org/10.1007/s10462-024-10729-y

function BSBOA = BinarySecretaryBirdOptAlg_tstf(feat,label,opts)
lb=0;                              
ub=1;
thres = 0.5;
HO = opts.Model;
N = opts.N;
max_Iter = opts.T;

% Objective function
fun = @FitnessFunction;

% Number of dimensions
dim = size(feat,2); 
%% INITIALIZATION
% Initial 
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end 

% Fitness
fit  = zeros(1,N);
fitB = inf;
Xnew = zeros(N,dim);

for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),HO);
  if fit(i) < fitB
    fitB = fit(i); 
    Xgb  = X(i,:);
  end
end

%% main loop
for t = 1:max_Iter
    CF = (1-t/max_Iter)^(2*t/max_Iter);
         
    %% The secretary bird's predation strategy
    for i = 1:N
        if t < max_Iter/3  % Secretary bird search prey stage
           Rn = size(X,1);
           X_random_1 = randi([1,Rn]);
           X_random_2 = randi([1,Rn]);
           R1 = rand(1,dim);
           X1 = X(i,:) + (X(X_random_1,:) - X(X_random_2,:)) .* R1; 
           X1 = max(X1,lb);
           X1 = min(X1,ub);
        elseif t > max_Iter/3 && t < 2*max_Iter/3  % Secretary bird approaching prey stage
           RB = randn(1,dim);
           X1 = Xgb + exp((t/max_Iter)^4) * (RB - 0.5) .* (Xgb - X(i,:));
           X1 = max(X1,lb);
           X1 = min(X1,ub);
        else       % Secretary bird attacks prey stage
           RL = 0.5 * Levy(dim);
           X1 = Xgb + CF * X(i,:).* RL;
           X1 = max(X1,lb);
           X1 = min(X1,ub);
        end

        T = taper_tf(X1,2,1); % T1
        B = double(rand(size(X1)) < T);
                
        f_newP1 = fun(feat,label,B,HO);
        if f_newP1 <= fit(i)
            X(i,:) = X1;
            fit(i) = f_newP1;
        end
    end
     
%% Secretary Bird's escape strategy
    r = rand;
    k = randperm(N,1);
    Xrandom = X(k,:);
    for i = 1:N
        if r < 0.5
            %% C1: Secretary birds use their environment to hide from predators
            RB = rand(1,dim);
            X2 = Xgb + (1-t/max_Iter) * (1-t/max_Iter) * (2*RB-1) .* X(i,:);% Eq.(5) S1
            X2 = max(X2,lb);
            X2 = min(X2,ub);
      
        else
            %% C2:  Secretary birds fly or run away from the predator
            K = round(1+rand(1,1));
            R2 = rand(1,dim);
            X2 = X(i,:)+ R2 .* (Xrandom - K .* X(i,:)); %  Eq(5) S2
            X2 = max(X2,lb);
            X2 = min(X2,ub);             
        end

        T = taper_tf(X2,2,1); % T1
        B = double(rand(size(X2)) < T);
        
        f_newP2 = fun(feat,label,B,HO); %Eq (6)
        if f_newP2 <= fit(i)
            X(i,:) = X2;
            fit(i) = f_newP2;
        end

        if fit(i) < fitB
            fitB = fit(i); 
            Xgb  = X(i,:);
        end
    
    
    end %

    best_so_far(t) = fitB;
    SBOA_curve = best_so_far;
    %average(t) = mean(fit);
    fprintf('\nIteration %d Best (BSBOA_TShaped)= %f',t,SBOA_curve(t));
    t=t+1;
end 

% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);

% Store results
BSBOA.sf = Sf; 
BSBOA.ff = sFeat; 
BSBOA.nf = length(Sf); 
BSBOA.c  = SBOA_curve;
BSBOA.f  = feat;
BSBOA.l  = label;
BSBOA.acc = 1- SBOA_curve(end);
end

%% Levy flight function
function o = Levy(d)
beta=1.5;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
u=randn(1,d)*sigma;v=randn(1,d);step=u./abs(v).^(1/beta);
o=step;
end