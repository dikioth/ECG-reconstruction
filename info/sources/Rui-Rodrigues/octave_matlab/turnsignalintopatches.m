function patches=turnsignalintopatches(signal,patchsize,jump)

[nrows ncols]=size(signal);



assert(nrows==1);

npatches=(ncols-patchsize)/jump+1;

patches=zeros(npatches,patchsize);

for i=1:npatches

  patches(i,:)=signal(1+(i-1)*jump:(i-1)*jump+patchsize);

end

end