function producebasicdata(val,folder,jump)

%val gets loaded with the instruction:' load a98m.mat '
%
%folder must be a string with the folder name for example 'a98' 




%%read 'config.txt' to get dimension of basic autoencoder dimension for the signals 
%%we are interessed on

 [nums,signals,dims]=lerconfigs(folder);

 makenetdimsandfilenames(folder,nums,signals,dims);


 base='../';

 base=strcat(base,folder);

 base=strcat(base,'/');

%%%%%%%%%%%%%



 %number of signals used to reconstruct target
 numbasic=nums(1);


 %basic é uma matriz com a ordem dos sinais(em val) que vamos usar para reconstruir 
 %target nos ultimos 30 seg

 %basic is a matrix with the input signals order(em val)

 basic=nums(2:numbasic+1);

%target is the order(in val) of target signal
 target=nums(numbasic+2);

 [nrows,ncols]=size(val);

 assert(ncols==75000);

 short=val(:,1:71250);








%%basic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 basicsignals=val(basic,:);

 newbasic=normalize_signals(basicsignals);


 basicauxallpatches=cell(numbasic,1);



 %some signals are subsampled

 newdims=zeros(numbasic+1,3);

 for i=1:numbasic+1

   for j=1:3

     if ((j==1)&&(dims(i,j)==63))

       newdims(i,j)=125;

     else
       newdims(i,j)=dims(i,j);

     end  
   end
 end



%to have a shorter aim patchdata file
%in case resp is one of the training signals 

 respsignal=0;


 for i=1:numbasic

   %resp must be subsampled

   if(strcmpi (signals{i,1},'resp')==1)

     assert(dims(i,1)==125);

     respsignal=i;

     basicauxallpatches{i,1}=turnsignalintopatches(newbasic(i,:), 2*dims(i,1),jump);

     basicauxallpatches{i,1}=basicauxallpatches{i,1}(:,1:2:250);

   else
     
     basicauxallpatches{i,1}=turnsignalintopatches(newbasic(i,:), newdims(i,1),jump);

     if(dims(i,1)==63)

       basicauxallpatches{i,1}=basicauxallpatches{i,1}(:,1:2:end);

     end

   end

 end



  %%if one of the signal is resp the other must start later 

  if(respsignal~=0)

    %(250-125)/5=25
    lag=26;

    for i=1:numbasic


      if(i~=respsignal)

	     assert(newdims(i,1)==125);

	     basicauxallpatches{i,1}=basicauxallpatches{i,1}(lag:end,:);

      end

    end

  end




%%these will be used to train the autoencoders 

  basicauxtrainingpatches=cell(numbasic,1);

 for i=1:numbasic

   basicauxtrainingpatches{i,1}=basicauxallpatches{i,1}(1:end-3750/jump,:);

  end


%basic training and critical(corresponds to the end of time, last x sec)  
%%data files


for i=1:numbasic

 filem=strcat(base,'patchdata_',signals{i,1},'.txt');
 writematriz(filem,basicauxtrainingpatches{i,1});

 filem=strcat(base,'criticaltime_',signals{i,1},'.txt');
 aux=basicauxallpatches{i,1}(end-3750/jump:end,:);
 writematriz(filem,aux); 

 %added lately
 filem=strcat(base,signals{i,1},'_allpatches.txt');
 writematriz(filem,basicauxallpatches{i,1});

 %filem=strcat(base,signals{i,1},'_allpatches.txt');
 %aux=basicauxallpatches{i,1};
 %writematriz(filem,aux); 


end

%%


%%aim

%debug
fprintf('%s %d\n','target signal number is ',target);



 aim=short(target,:);




 newaim=normalize_signals(aim);

 %if aim is 'resp', 'cvp', 'abp' or 'art' it will be subsampled
 if(strcmpi(signals{numbasic+1,1},'resp')==1)

   assert(dims(numbasic+1,1)==125);
  
   patchsizeaim=375;

   patchesaim=turnsignalintopatches(newaim,patchsizeaim,jump);

   patchesaim=patchesaim(:,1:3:patchsizeaim);

 elseif((strcmpi(signals{numbasic+1,1},'cvp')==1)...
	||(strcmpi(signals{numbasic+1,1},'abp')==1)...	
	||(strcmpi(signals{numbasic+1,1},'art')==1)...
	||(strcmpi(signals{numbasic+1,1},'pleth')==1))

   assert(dims(numbasic+1,1)==63);

   patchsizeaim=125;

   patchesaim=turnsignalintopatches(newaim,patchsizeaim,jump);

   patchesaim=patchesaim(:,1:2:patchsizeaim);   

 else

   patchesaim=turnsignalintopatches(newaim,dims(numbasic+1,1),jump);

 end


 if(respsignal~=0)

   patchesaim=patchesaim(lag:end,:);

 end


 aimfile=strcat(base,'patchdata_aim.txt');

 writematriz(aimfile,patchesaim);


end


