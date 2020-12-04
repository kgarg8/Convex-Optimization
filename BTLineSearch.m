function t = BTLineSearch(func, x, z, alpha, beta, dx, dfx)
% Function to calculate step length [known as BackTracking Line Search]
% Refer BV04, Algorithm 9.2
% Inputs:
%   func    : Objective function
%   x       : Decision variable value
%   z       : exp(A*x+b)
%   alpha   : constant, characteristic of BTLineSearch
%   beta    : constant, characteristic of BTLineSearch
%   dx      : Newton step at given x
%   dfx     : Gradient of objective function at given x
% Outputs:
%   t : Step length
t = 1;
while func(x + t*dx) > func(x) + alpha*t*dfx(z,x)'*dx
    t = beta*t;
end
