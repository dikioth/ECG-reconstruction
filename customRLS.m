function bi = customRLS(xT, x1, x2, N)
% x2 = ECG_AVR
% xT = ECG_II


% number of iterations
Nsim=1000; 

% number of unknowns, length of theta vector
p=N; % ai and bi

% We assume that x[n] = theta_1 x[n-1] + \theta_2 x[n-2]


x = x2;

% RLS forgetting factor
mylambda=0.95;


% Initilizations
mytheta=0.2*ones(p,1);
P=100*eye(p);

% keep theta and error in an array to plot later
eA=zeros(Nsim,1);
thetaA(:,1)=mytheta;
for i=1:Nsim-1-p
    % new row for $H$
    h=x(i+p-1:-1:i,1);
    % error (this is prediction error, not the ls error )
    e=xT(i+p,1)-h.'*mytheta;
    % update kalman gain
    K= P*h/(mylambda^i +h.'*P*h); 
    %%%% TODO %%%%%%%%%%%%%%%%%%
    % update theta 
    mytheta=mytheta + K * e; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % update P
     P = (eye(p)- K*h.')*P;
       % store the values to plot later
    thetaA(:,i+1)=mytheta;
    eA(i,1)=e;  
end


bi = mytheta;

% Below code is written for p=2
% hf1=figure;
% % Plot RLS filter coefficients, i.e. theta_1 and theta_2
% h11=plot(thetaA(1,:)',':','Linewidth',2);
% hold on
% h12=plot(thetaA(2,:)',':','Linewidth',2);
% hold on
% % Plot the parameters of the process
% h21=plot(ones(Nsim+1,1)* a1, '-k','Linewidth',2);
% h22=plot(ones(Nsim+1,1)* a2, '--k','Linewidth',2);
% xlim([1,Nsim+1]);
% hxlabel=xlabel('Iteration');
% hylabel=ylabel('Filter Coefficients');
% legend([h11 h12 h21 h22], 'c1-Estimate', 'c2-Estimate', 'c1-True','c2-True','Location', 'best');
% grid on;
% set(gca,'FontSize',16);
% set(hxlabel,'FontSize', 18);
% set(hylabel,'FontSize', 18);
% set(gca,'FontSize',16);
% epsName=sprintf('figAR2_RLS.eps');
% set(hf1, 'PaperPositionMode', 'auto');
% saveas(hf1,epsName,'epsc')

end
% %Look at the error 
% figure
% plot(eA.^2)
