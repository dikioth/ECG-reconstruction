function produceother(val,folder,jump,othersignalsorderinbasic,otherpatches_size,othername)

%val is the matrix that contains all signals you get it doing:' load a98m.mat '
%
%folder must be a string with the folder name for example 'a98' 
%
%othersignalsorderinbasic is a vector wich contains the order of the 
%signals in the file config.txt wich we choose to belong to other
%
%otherpatches_size is a vector with the patchsize for each of 
%the signals that will belong to other it should be a multiple of the 
%patch size that appears in config.txt
%
%othername is a string with the name of 'other'



%%read 'config.txt' to get dimension of basic autoencoder dimension for the signals 
%%we are interessed on

 [nums,signals,dims]=lerconfigs(folder);

 %makenetdimsandfilenames(folder,nums,signals,dims);




 base='../';

 base=strcat(base,folder);

 base=strcat(base,'/');

%%%%%%%%%%%%%


 %numero de sinais usados para reconstruir target
 numbasic=nums(1);


 %basic é uma matriz com a ordem dos sinais(em val) que vamos usar para reconstruir 
 %target nos ultimos 30 seg
 basic=nums(2:numbasic+1);


 %patchsize=dims(1,1);

 [nrows,ncols]=size(val);

 assert(ncols==75000);


%%basic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


 basicsignals=val(basic,:);

 newbasic=normalize_signals(basicsignals);




 other=newbasic(othersignalsorderinbasic,:);

 numother=size(other,1);




 %others normalized during training time
 %otherstraining=newothers(:,1:71250);

 %numallpatches=ceil((ncols-patchsize)/jump)+1;

 otherauxallpatches=cell(numother,1);


 respsignal=0;

 for i=1:numother

   %some signals will  be subsampled

   order=othersignalsorderinbasic(i);

   if(strcmpi(signals{order,1},'resp')==1)

     assert(otherpatches_size(i)==125);

     respsignal=i;


     otherauxallpatches{i,1}=turnsignalintopatches(other(i,:),...
						   2*otherpatches_size(i),jump);

     otherauxallpatches{i,1}=otherauxallpatches{i,1}(:,1:2:250);

   elseif(otherpatches_size(i)==63)
     
     otherauxallpatches{i,1}=turnsignalintopatches(other(i,:), 125,jump);

     otherauxallpatches{i,1}=otherauxallpatches{i,1}(:,1:2:end);
     
   elseif(otherpatches_size(i)==189)

     otherauxallpatches{i,1}=turnsignalintopatches(other(i,:), 375,jump);

     otherauxallpatches{i,1}=otherauxallpatches{i,1}(:,1:2:end);
     %size2 is 188!=189 must add first column  

     ext=zeros(size(otherauxallpatches{i,1},1),1);
     ext(1,1)=otherauxallpatches{i,1}(1,1);
     ext(2:end,1)=otherauxallpatches{i,1}(1:end-1,end);
     
     otherauxallpatches{i,1}=[ext,otherauxallpatches{i,1}];

   %%not necessary subsampling
   else

     otherauxallpatches{i,1}=turnsignalintopatches(other(i,:),...
						   otherpatches_size(i),jump);




   end


  end



  %%if one of the signal is resp the other must start later 

  if(respsignal~=0)

    %(250-125)/5=25
    lag=26;

    for i=1:numother


      if(i~=respsignal)

	     assert((otherpatches_size(i)==125)||(otherpatches_size(i)==63));

	     otherauxallpatches{i,1}=otherauxallpatches{i,1}(lag:end,:);

      end

    end

  end


 otherallpatches=otherauxallpatches{1,1};


 for i=2:numother

   otherallpatches=[otherallpatches,otherauxallpatches{i,1}];

 end

 filem=strcat(base, othername, '_allpatches.txt');

 writematriz(filem,otherallpatches);



%%these will be used to train the mlp4 net


 othertraining=otherallpatches(1:end-3750/jump,:);

 filem=strcat(base,'patchdata_',othername, '.txt');

 writematriz(filem,othertraining);


%%critical time


 othercriticaltime=otherallpatches(end-3750/jump:end,:);

 filem=strcat(base,'criticaltime_',othername, '.txt');

 writematriz(filem,othercriticaltime);



%%

  makeotherconfigfile(val,folder,jump,othersignalsorderinbasic,...
			     otherpatches_size,dims,othername);



end



function makeotherconfigfile(val,folder,jump,othersignalsorderinbasic,...
			     otherpatches_size,dimsotherbasic,othername)





  numother=size(othersignalsorderinbasic,2);

  assert(size(otherpatches_size,2)==numother);

  
  dimsotherbasicefectiv=dimsotherbasic(othersignalsorderinbasic,:);
  

  %compute dims




  factor=ceil(otherpatches_size(1)/dimsotherbasicefectiv(1,1));




  %assert rest is zero

  assert(factor*dimsotherbasicefectiv(1,1)==otherpatches_size(1));


  %assert for each of the other signal factor is the same

  for i=2:numother

    assert(factor*dimsotherbasicefectiv(i,1)==otherpatches_size(i));

  end


  dims=zeros(1,numother);

  dims(1)=sum(otherpatches_size,2);


  for i=2:3

    dims(i)=factor*sum(dimsotherbasicefectiv(:,i));

  end


  base='../';

  base=strcat(base,folder);

  filename=strcat(base,'/',othername,'.txt');

  pid = fopen (filename, 'w');


  fprintf(pid, '1\n%d\n%d\n%d\n%d\n',dims(1),dims(2),dims(3),jump);

  texto=makefilenames(folder,othername);

  fprintf(pid,'%s',texto);


  fclose(pid);


end