function result = getSweep(pn, Mintervall, Nintervall)
%getSweep: The function performs a sweep for diferrent values of M and N.
% result = getSweep(pn, [Mstart Mend], [Nstart, Nend]);

p = getpatient(pn);

iters = max([max(Mintervall), max(Nintervall)]);
Nvect = zeros(iters,1);
Mvect = zeros(iters,1);
Q1v = zeros(iters, 1);  
Q2v = zeros(iters, 1); 

i = 1;
for n = Nintervall
    for m = Mintervall
        ci = customADAM(p.xTzm, p.x1zm, m, p.x2zm, n);
        xhat = getReconstruction(ci, p.x1zm, m, p.x2zm, n) + p.xTmean;
        [Q1, Q2] = getPerformance(p.xTm, xhat);
        
        Nvect(i) = n;
        Mvect(i) = m;
        Q1v(i) = Q1;
        Q2v(i) = Q2;
        
        i = i + 1;
        fprintf('Patient: %d. N: %d, M: %d\n', pn, n, m);
    end
end

% Finding max values
result.maxQ1 = Q1v(Q1v == max(Q1v)); 
result.maxQ1M = Mvect(Q1v == max(Q1v));
result.maxQ1N = Nvect(Q1v == max(Q1v));

result.maxQ2 = Q2v(Q2v == max(Q2v));
result.maxQ2M = Mvect(Q2v == max(Q2v));
result.maxQ2N = Nvect(Q2v == max(Q2v));

result.Nvect = Nvect;
result.Mvect = Mvect;
result.Q1v = Q1v;
result.Q2v = Q2v;
end