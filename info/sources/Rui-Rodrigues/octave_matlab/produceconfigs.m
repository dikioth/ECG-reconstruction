function [ordersignals, dimensions]=produceconfigs(folder,nameofinputsignals,nameaim)

%folder: exemple 'c16'
%val is the matrix obtained with 'load c16m.mat'
%nameofinputsignals is a cell array (1,numberofinputsignals) with the names of the signals we want to use to reconstruct target signal





  base='../';

  base=strcat(base,folder);

  configfile=strcat(base,'/config.txt');

  infofile=strcat(base,'/',folder,'m.info');


  %debug
  %fprintf(1,'config file is %s \n info file is %s \n',configfile,infofile);



  %get the order of input signals and aim in val


  numberofinputsignals=size(nameofinputsignals,2);

  nameofinputsignals{1,numberofinputsignals+1}=nameaim;
  

  ordersignals=findsignalsininfo(nameofinputsignals,numberofinputsignals+1,infofile);


  dimensions=zeros(numberofinputsignals+1,1);



  %dimensions for each input signal autoencoder

  for i=1:numberofinputsignals

    if((isecg(nameofinputsignals{1,i}))||(strcmpi (nameofinputsignals{1,i}, 'icp')==1))
	
      dimensions(i,1)=125;

      dimensions(i,2)=150;

      dimensions(i,3)=175;      
    
    elseif(strcmpi (nameofinputsignals{1,i}, 'resp')==1)

      dimensions(i,1)=125;

      dimensions(i,2)=75;

      dimensions(i,3)=100;      


    else

      dimensions(i,1)=63;

      dimensions(i,2)=75;

      dimensions(i,3)=100;      

    end

  end


  %dimensions for target signal autoencoder



  if (strcmpi (nameaim,'resp')==1)

    dimensions(numberofinputsignals+1,1)=125;

    dimensions(numberofinputsignals+1,2)=100;

    dimensions(numberofinputsignals+1,3)=35;  


  elseif(strcmpi (nameaim,'icp')==1)

    dimensions(numberofinputsignals+1,1)=125;

    dimensions(numberofinputsignals+1,2)=100;

    dimensions(numberofinputsignals+1,3)=60;  


  elseif (isecg(nameaim))
  
    dimensions(numberofinputsignals+1,1)=125;

    dimensions(numberofinputsignals+1,2)=100;

    dimensions(numberofinputsignals+1,3)=75;  
 

 
  else

    dimensions(numberofinputsignals+1,1)=63;

    dimensions(numberofinputsignals+1,2)=70;

    dimensions(numberofinputsignals+1,3)=50;  


  end



  %write everything on config file

  fid = fopen (configfile,'w');

  %debug
  %fid=1;

  %write numberofinputsignals, for each inputsignal its order in val
  %and aim order also  

  fprintf(fid,'%d',numberofinputsignals);

  for i=1:numberofinputsignals+1

    fprintf(fid,' %d',ordersignals(i));

  end

  fprintf(fid,'\n'); 

  for i=1:numberofinputsignals+1

    fprintf(fid,'%s %d %d %d\n',nameofinputsignals{1,i},dimensions(i,1),dimensions(i,2),dimensions(i,3));

   end

   fclose(fid);

end


function res=isecg(name)

  if(strcmpi (name,'II'))

    res=true;

  elseif(strcmpi (name,'V'))

    res=true;

  elseif(strcmpi (name,'I'))

    res=true;

  elseif(strcmpi (name,'III'))

    res=true;

  elseif(strcmpi (name,'V'))

    res=true;

  elseif(strcmpi (name,'AVR'))

    res=true;

  elseif(strcmpi (name,'AVF'))

    res=true;

  elseif(strcmpi (name,'MCL'))

    res=true;

  else

    res=false;

  end

end

function orders=findsignalsininfo(inputsignalsandaim,numberofinputsandaim,infofile)

%inputsignalsandaim is a cell array (numberofinputandaim,1) with the names of inputsignals and aim in the end
%configfile is a string wich includes the path to the file

  fid=fopen(infofile);


  % lin will be the number of lines in the the info file

  lin=0;

  while (~ feof (fid) )

    lin=lin+1;
    fgetl (fid);

  end


  totnumberofsignals=lin-8;

  frewind (fid)


  %trash

  for i=1:5
    fgetl (fid); 
  end

  signalsname=cell(lin-8,1);
  signalsnum=zeros(lin-8,1);

  for i=1:lin-8;

    signalsnum(i)=fscanf (fid,'%10d' ,1);

    signalsname{i,1}=fscanf (fid, '%s',1);

    %trash

    for j=1:2
      num=fscanf (fid,'%10f' ,1);
    end

    aux=fscanf (fid, '%s',1);

  end


  orders=zeros(1,numberofinputsandaim);


  for i=1:numberofinputsandaim

    res=strcmpi(inputsignalsandaim{1,i},signalsname);

    if(~any(res))

      printf('signal %s is not present!',inputsignalsandaim{1,i});

      return;

    else

      orders(i)=find(res);      
      
    end

  end

end