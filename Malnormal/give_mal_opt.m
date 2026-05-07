function [malA,exitflag] = give_mal_opt(A,n)
% Input a square nxn matrix A as well as n 
% Returns malA, the constant of malnormality of A, using optimization along with the exit flag
% so that we know if the algorithm converged to a minimum. 

% Requires license for MATLAB Optmization Toolbox and the function optimization_constraint_iso.m

% dimension of H_n0
k = ((1/2)*n*(n+1))-1;

% create function, this is the squared norm of the commutator
fun = @(x) trace((A*phi_iso2(x,n)-phi_iso2(x,n)*A)'*(A*phi_iso2(x,n)-phi_iso2(x,n)*A));

% % initial point
x0 = rand(k,1);
x0 = x0/norm(x0);


% customizing optimization problem
% options = optimoptions('fmincon', 'Algorithm','interior-point','MaxFunEvals',Inf,'MaxIter',1500,'SpecifyConstraintGradient',true);
options = optimoptions('fmincon', 'Algorithm','interior-point','MaxFunEvals',Inf,'MaxIter',1500,'SpecifyConstraintGradient',true, OptimalityTolerance=1e-10, FunctionTolerance=1e-20, StepTolerance=1e-20, ConstraintTolerance=1e-10);
% for ginibre dim=30 we added ('HessianApproximation','lbfgs') to the above line

% optimization problem
% using interior point algorithm
[x,feval,exitflag] = fmincon(@(x)fun(x),x0,[],[],[],[],[],[],@(x)optimization_constraint_iso(x),options);

% display(x);

malA = sqrt(feval);

end