function signal=saturate(othername,folder,maximumvalue,minimumvalue)

%% othername='otherecgpleth', folder='c12', maximumvalue=4199, minimumvalue=0
%
%%
%%reads missing signal file associated to 'othername' and writes back the sinal
%% in the same file but with all values superior to 'value' are reduced to 'value' 


 base='../';

 originalsignalfile=strcat(base,folder,'/missingsignal_',othername,'_aim.txt');

 saturatedsignalfile=strcat(base,folder,'/saturatedsignal_',othername,'_aim.txt');

 fprintf(1,'%s %s \n ','reading from file: ',originalsignalfile);

 fid = fopen (originalsignalfile, 'r');

 signal=fscanf(fid,'%f',3750);

 fclose (fid);

 for i=1:3750

   if(signal(i)>maximumvalue)
     signal(i)=maximumvalue;
   elseif(signal(i)<minimumvalue)
     signal(i)=minimumvalue;
   end

 end


 fid = fopen (saturatedsignalfile, 'w');

 for i=1:3750

   fprintf(fid,'%f \n',signal(i));

 end

 fclose(fid);

end



