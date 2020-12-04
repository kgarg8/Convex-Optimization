% Newton's Method for finding the optimal value that minimzes a function


% Load variables, define parameters
load('A8_7_data.mat');      % load A, b
x = zeros(size(A,2),1);     % Initialize decision variable
alpha = 0.01;   % Parameter for BT Line Search
beta = 0.5;     % Parameter for BT Line Search
eps = 1e-10;    % tolerance


% Define function, gradient & hessian
func = @(x) 0.5*x'*x + log_sum_exp(A*x+b);  % Objective Function
dfx = @(z, x) x + (A'*z)/ sum(z);   % Gradient
d2fx = @(z) eye(size(A,2)) + A'*((diag(z)/sum(z)) - (z*z'/square(sum(z))))*A; % Hessian


% Variables tracked during loop
func_vals = zeros(100); % store the func iterates in an array
func_vals(1) = func(x);
k=1 % iteration


% Heart of Newton's Method
% Refer BV04, Algorithm 9.5
tic %  Clock for Newton Method convergence
while true
    z = exp(A*x+b);
    dx = -inv(d2fx(z))*dfx(z,x);    % (1) Newton step
    newton_decrement = -dfx(z,x)'*dx;   % (2) Newton decrement
    if newton_decrement/2 <= eps
        break
    end
    t = BTLineSearch(func, x, z, alpha, beta, dx, dfx); % (3) Calculate step length
    x = x + t*dx;       % (4) Update x
    func_vals(k+1) = func(x);
    k = k+1
end
func(x)
toc % Time elapsed for Newton method convergence


% Plot f(x^k) - p^* on logscale
x_axis = 1:k
y_axis = func_vals(1:k) - func_vals(k)
semilogy(x_axis, y_axis)
xlabel('iterations')
ylabel('f(x^k) - optval (logscale)')
title('Convergence of Newton Method')
grid on