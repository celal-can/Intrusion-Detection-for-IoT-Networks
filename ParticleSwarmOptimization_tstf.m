% This algorithm is adapted from Too, Jingwei, et al. 
% "EMG Feature Selection and Classification Using a Pbest-Guide Binary Particle Swarm Optimization." 
% Computation, vol. 7, no. 1, MDPI AG, Feb. 2019, p. 12, doi:10.3390/computation7010012.

function BPSO = ParticleSwarmOptimization_tstf(feat,label,opts)
% Parameters
lb    = 0; 
ub    = 1;
thres = 0.5;           
Vmax  = (ub - lb) / 2;  % Maximum velocity 
N = opts.N;
max_Iter = opts.T;
c1 = opts.c1;  % cognitive factor
c2 = opts.c2;  % social factor 
w = opts.w;    % inertia weight
HO = opts.Model;

% Objective function
fun = @FitnessFunction; 

% Number of dimensions
dim = size(feat,2); 

% Initial 
X   = zeros(N,dim); 
V   = zeros(N,dim); 
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
  % Gbest update
  if fit(i) < fitG
    Xgb  = X(i,:); 
    fitG = fit(i);
  end
end
% PBest
Xpb  = X; 
fitP = fit;
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 1;

% Iterations
while t <= max_Iter
  for i = 1:N
    for d = 1:dim
      r1 = rand();
      r2 = rand();
      % Velocity update (2a)
      VB = w * V(i,d) + c1 * r1 * (Xpb(i,d) - X(i,d)) + ...
        c2 * r2 * (Xgb(d) - X(i,d));
      % Velocity limit
      VB(VB > Vmax) = Vmax;  VB(VB < -Vmax) = -Vmax;
      V(i,d) = VB;
      % Position update (2b)
      X(i,d) = X(i,d) + V(i,d);
    end
    
    T = taper_tf(X(i,:),2,1); % T1
    B = double(rand(size(X(i,:))) < T);

    % Fitness
    fit(i) = fun(feat,label,B,HO);
    
    % Pbest update
    if fit(i) < fitP(i)
      Xpb(i,:) = X(i,:); 
      fitP(i)  = fit(i);
    end
    % Gbest update
    if fitP(i) < fitG
      Xgb  = Xpb(i,:);
      fitG = fitP(i);
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (BPSO)= %f',t,curve(t))
  t = t + 1;
end

% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf); 

% Store results
BPSO.sf = Sf; 
BPSO.ff = sFeat;
BPSO.nf = length(Sf);
BPSO.c  = curve;
BPSO.f  = feat;
BPSO.l  = label;
BPSO.acc = 1 - curve(end);
end