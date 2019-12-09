function res=computecorrelation(signal1,signal2)

assert((size(signal1,1)==1)&&(size(signal2,1)==1));

assert(size(signal1,2)==size(signal2,2));

l=size(signal1,2);

n1=(signal1-mean(signal1,2))/std(signal1);

n2=(signal2-mean(signal2,2))/std(signal2);

res=sum(n1.*n2,2)/l;

end