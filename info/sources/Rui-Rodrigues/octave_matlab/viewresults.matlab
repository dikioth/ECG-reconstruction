function [patchdata, fwdallpatches, reclongsignal]=viewresults(folder,val,othername,ordmissignal)

directoria=strcat('../',folder);



patchdatafile=strcat(directoria,'/patchdata_aim.txt');

patchdata=readmatriz(patchdatafile);




fwdallpatchesfile=strcat(directoria,'/fwd_allpatches_mlp4layers_',othername,'_aim.txt');

%printf('%s %s \n','reading from file: ',fwdallpatchesfile);

fwdallpatches=readmatriz(fwdallpatchesfile);



reclongsignalfile=strcat(directoria,'/reconstructed_long_signal_',othername,'_aim.txt');

%printf('%s %s','reading from file: ',reclongsignalfile);

reclongsignal=readmatriz(reclongsignalfile);

assert(size(reclongsignal,1)==1);

lengthrec=size(reclongsignal,2);



missignal=val(ordmissignal,:);

missignal=missignal(1:end-3750);


tmissignal=missignal(1,end-lengthrec+1+3750:end);

[a,b]=score(tmissignal,reclongsignal(1,1:end-3750));

fprintf(1,'%s %f %f \n', 'before denormalization score is ', a,b);


meanaim=mean(missignal,2);

stdaim=std(missignal);

reclongsignal=stdaim*reclongsignal+meanaim;

[a,b]=score(tmissignal,reclongsignal(1,1:end-3750));

fprintf(1,'%s %f %f \n', 'after denormalization score is ', a,b);





missingsignal=reclongsignal(1,end-3750+1:end);

%file to upload to physionet

missingsignalfile=strcat(directoria,'/missingsignal_',othername,'_aim.txt');

fid = fopen (missingsignalfile, 'w');

fprintf(fid,'%10.7f \n',missingsignal);

fclose (fid);

patchsize=size(patchdata,2);

s=123;
x=1:patchsize;
plot(x,patchdata(s,x),'k',x,fwdallpatches(s,x),'r');

