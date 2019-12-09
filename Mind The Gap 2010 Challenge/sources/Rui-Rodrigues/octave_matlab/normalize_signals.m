function newsignals=normalize_signals(signals)

%%from the signals matrix  builds a newmatrix where each line is 
%normalized ie average remove and then the result is divided by std



numsignals=size(signals,1);


%average=0
%std=1

Smean=mean(signals,2);

for i=1:numsignals;

  signals(i,:)=signals(i,:)-Smean(i);

end


std=sqrt(mean(signals.^2,2));


for i=1:numsignals;

  signals(i,:)=signals(i,:)/std(i);

end

newsignals=signals;

end