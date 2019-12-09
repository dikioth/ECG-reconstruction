function makenetdimsandfilenames(folder,nums,signals,dims)

%folder como 'b31'
%numother e o numero de sinais usados para reconstruir o target

%nums e o que está na 1 linha de config

%signals e o nome dos sinais 'other' e 'aim'--- cellarray

%dims e uma matriz 

  %[nums,signals,dims]=lerconfigs(folder);

  numother=nums(1);



  jump=5;

  nome=strcat('../',folder);

  for i=1:numother

    aux='-';
    aux=signals{i,1};

    filename=strcat(nome,'/',aux,'.txt');

    disp(filename);

    fid = fopen (filename, 'w');

    fprintf (fid, '\n%d\n%d\n%d\n%d\n',dims(i,1),dims(i,2),dims(i,3),jump);

    texto=makefilenames(folder,aux);

    fprintf (fid,'%s',texto);

    fclose (fid);

  end



%%
%aim
%%


aim='aim';

ninputs=dims(numother+1,1);

nhid0=dims(numother+1,2);

nhid1=dims(numother+1,3);

filename=strcat(nome,'/',aim,'.txt');

disp(filename);

fid = fopen (filename, 'w');

fprintf (fid, '%d\n%d\n%d\n%d\n%d\n',1,ninputs,nhid0,nhid1,jump);

texto=makefilenames(folder,aim);

fprintf (fid,'%s',texto);

fclose (fid);


end

