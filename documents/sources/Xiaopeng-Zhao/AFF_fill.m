%AFF_fill.m
%   Fill gaps by Averaged Feedback of a Feedforward neural network model
%   In response to 2010 physionet challenge -- Mind the Gap, the software
%   fills signal gaps. The program is based on a feedforward neural network
%   model. We adopt the following techniques to improve the prediction 
%   accuracy. 
%       1. Parameters of the network are optimized for better performance.
%       2. Mix of short- and long-term memory.
%       3. Iteration of feedback on the results of the feedforward model.
%       4. Average of the results from the feedback iterations.
%   The code has been tested on Matlab 2008a and Matlab 2010a.
%
%   Copyright (C) 2010  
%   Xiaopeng Zhao <xzhao9@utk.edu>
%   Nonlinear Biodynamics Laboratory
%   Department of Biomedical Engineering
%   University of Tennessee
%   Knoxville, TN 37996
%
%   This software is released under the terms of the GNU General
%   Public License (http://www.gnu.org/copyleft/gpl.html).

        
clear all
patientclass='c';
filepath=['../physionetdata/set-' patientclass '/'];

numofiteration=3;
datapoints=10*3750;
myepochs=40;
mydelay=[0:5,10,20,50,100,200];
mygoal=0;
myperfFcn='msne';
mylayers=[10,1];
mytransferfcn={'tansig' 'purelin' };

yp{numofiteration}=[];
yrec{numofiteration}=[];

for pid=0:99
    if pid < 10
        patientid = [ '0' num2str(pid) ];
    else
        patientid = num2str(pid);
    end
    filenm=[patientclass patientid 'm.mat'];
    load([filepath,filenm]);
    disp(['---------------------------------I am working on ', filenm]);
    %identify input channels and target channel
    totalSignalNum = size(val, 1 );
    missing_channel=[];
    input_channels=[];
    for i = 1 : totalSignalNum
        if val(i, end-3750+1:end) == 0
            missing_channel=[missing_channel,i];
        else
            input_channels=[input_channels,i];
        end
    end
    %if multiple channels have missing data, we check if the channel is
    %really a missing channel.
    miss_id=1;
    for i=1:length(missing_channel)
        if val(missing_channel(i),1:1000)==0
        else
            miss_id=i;
        end
    end
    missing_channel=missing_channel(miss_id);
    %input and output data
    inpdata=val(input_channels,:); %
    outdata=val(missing_channel,:); %
    inpdata=removeconstantrows(inpdata);
    [inpdata, inps] = mapminmax(inpdata);
    [outdata, outs] = mapminmax(outdata);
    %augment the input data to consider delays
    [inputnum,totlen] = size(inpdata);
    dmax=max(mydelay); dnum=length(mydelay);
    x=zeros(dnum*inputnum,totlen-dmax);
    for i=1:length(mydelay)
        di=mydelay(i);
        pos=(i-1)*inputnum+(1:inputnum);
        x(pos,:)=inpdata(:,dmax+1-di:totlen-di);
    end
    y=outdata(dmax+1:end);
    
    %feebforward prediction
    indx=size(x,2)-datapoints+1-3750:size(x,2)-3750;
    p = x(:,indx);
    t = y(:,indx);
    clear mynet
    mynet = newff(p,t, mylayers, mytransferfcn );
    mynet.performFcn=myperfFcn;
    mynet.trainParam.show = 5;
    mynet.trainParam.showWindow = false;
    mynet.trainParam.epochs = myepochs;
    mynet.trainParam.goal=mygoal;
    [mynet,tr] = train(mynet,p,t);
    xp = x(:,end-3750+1:end);
    yp{1} = sim( mynet, xp );
    yrec{1}=mapminmax('reverse',yp{1}, outs)';
    
    %feedback
    indx=size(x,2)-datapoints+1:size(x,2);
    p = x(:,indx);
    for i=2:numofiteration
        y(end-3750+1:end)=yp{i-1};
        t = y(:,indx);
        clear mynet
        mynet = newff(p,t, mylayers, mytransferfcn );
        mynet.performFcn=myperfFcn;
        mynet.trainParam.epochs = myepochs;
        mynet.trainParam.goal=mygoal;
        mynet.trainParam.show = 5;
        mynet.trainParam.showWindow = false;
        [mynet,tr] = train(mynet,p,t);
        yp{i} = sim(mynet, xp);
        yrec{i}=mapminmax('reverse',yp{i}, outs)';
        
    end
    ydata=cell2mat(yrec);
    yavg=mean(ydata,2);
    
    save([filenm(1:3),'.predicted'],'yavg','-ascii');
end