
function ab = adamOpt(xT, x1, x2, N, M)

% LMS filter order
NM = N+M;
maxNM = max(N,M);
% LMS filter parameters
mn = zeros(NM,1);
vn = zeros(NM,1);

beta1 = 0.89;
beta2 = 0.999;
alpha = 0.0021;
epsilon = 1e-5;

% LMS filter initilization
h=0.2*ones(NM,1);

% number of iterations
Nsim=length(xT); 

for i=maxNM:Nsim
    %observations for this step
    d = [xT(i:-1:i-N+1,1); xT(i:-1:i-M+1,1)];
    y = [x1(i:-1:i-N+1,1); x2(i:-1:i-M+1,1)];
    
    gn = (h.'*h).*y - h.'*d;
    % filter update
    mn = beta1*mn + (1-beta1)*gn;
    vn = beta2*vn + (1-beta2)*gn.^2;
    my_mn = mn/(1-beta1^i);
    my_vn = vn/(1-beta2^i);
    h(:,1)= h(:,1) + alpha*my_mn./(sqrt(my_vn)+epsilon);
end
ab = h;
end
