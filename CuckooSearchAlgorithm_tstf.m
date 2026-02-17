% This algorithm is adapted from X. -S. Yang and Suash Deb, 
% "Cuckoo Search via Lévy flights," 
% 2009 World Congress on Nature & Biologically Inspired Computing (NaBIC), Coimbatore, India, 2009, pp. 210-214, doi: 10.1109/NABIC.2009.5393690.


function CS = CuckooSearchAlgorithm_tstf(feat,label,opts)

% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
Pa    = 0.25;   % discovery rate
alpha = 1;      % constant
beta  = 1.5;    % levy component

HO = opts.Model;
N = opts.N;
max_Iter = opts.T;

% Objective function
fun = @FitnessFunction; 

% Number of dimensions
dim = size(feat,2); 

% Initial 
X   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  

% Fitness
fit  = zeros(1,N);
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),HO); 
  % Best cuckoo nest
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end

% Pre
Xnew = zeros(N,dim);
curve = zeros(1,max_Iter); 
curve(1) = fitG;
t = 1;  

% Iterations
while t <= max_Iter
  % {1} Random walk/Levy flight phase
  for i = 1:N
    % Levy distribution
    L = LevyDistribution(beta,dim);
    for d = 1:dim
      % Levy flight (1)
      Xnew(i,d) = X(i,d) + alpha * L(d) * (X(i,d) - Xgb(d));
    end
    
    T = taper_tf(Xnew(i,:),2,1); % T1
    B = double(rand(size(Xnew(i,:))) < T);
  end
  % Fintess
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,B,HO); 
    % Greedy selection
    if Fnew <= fit(i)
      fit(i) = Fnew;
      X(i,:) = Xnew(i,:);
    end
  end
  % {2} Discovery and abandon worse nests phase
  Xj = X(randperm(N),:); 
  Xk = X(randperm(N),:);
  for i = 1:N 
    Xnew(i, :) = X(i,:);
    r          = rand();
    for d = 1:dim
      % A fraction of worse nest is discovered with a probability
      if rand() < Pa
        Xnew(i,d) = X(i,d) + r * (Xj(i,d) - Xk(i,d));
      end
    end
    T = taper_tf(Xnew(i,:),2,1); % T1
    B = double(rand(size(Xnew(i,:))) < T);
  end
  % Fitness
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,B,HO); 
    % Greedy selection
    if Fnew <= fit(i)
      fit(i) = Fnew;
      X(i,:) = Xnew(i,:);
    end
    % Best cuckoo
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (BCSA)= %f',t,curve(t));
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
CS.sf = Sf; 
CS.ff = sFeat; 
CS.nf = length(Sf); 
CS.c  = curve;
CS.f  = feat;
CS.l  = label;
CS.acc = 1- curve(end);
end

%// Levy Flight //
function LF = LevyDistribution(beta,dim)
% Sigma 
nume  = gamma(1 + beta) * sin(pi * beta / 2); 
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
sigma = (nume / deno) ^ (1 / beta); 
% Parameter u & v 
u = randn(1,dim) * sigma; 
v = randn(1,dim);
% Step 
step = u ./ abs(v) .^ (1 / beta);
LF   = 0.01 * step;
end
