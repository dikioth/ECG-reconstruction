function producedata(val,other,target,folder)

%val is the matrix loaded by an instruction like ' load a98m.mat '
%
%other is a cell array (size (1,n) )of strings that can be defined like {'resp','cvp', 'II'} 
%
%target is the string with the name of target signal
%
%folder is a string with the folder name, for example 'a98' 


  numother=size(other,2);


  

  [ordersignals, dimensions]=produceconfigs(folder,other,target);

 

  %jump is the gap between the beginnings of two consecutive training patches
  jump=5;

  producebasicdata(val,folder,jump);


  %othername

  othername='other';

  for i=1:numother

    othername=strcat(othername,other{1,i});

  end


  %otherpathes_size

  if(strcmpi (target,'resp')==1)

    otherpatches_size=(3*dimensions(1:end-1,1))';

  else

    otherpatches_size=(dimensions(1:end-1,1))';    

  end


  produceother(val,folder,jump,[1:numother],otherpatches_size,othername);

end
