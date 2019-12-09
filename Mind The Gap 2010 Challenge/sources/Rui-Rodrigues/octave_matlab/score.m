function [a,correlation]=score(signal1,signal2)

assert((size(signal1,1)==1)&&(size(signal2,1)==1));

assert(size(signal1,2)==size(signal2,2));

l=size(signal1,2);

correlation=computecorrelation(signal1,signal2);

dif=sum((signal1-signal2).^2,2);

a=1-dif/(l*(std(signal1)^2));

