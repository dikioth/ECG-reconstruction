
function [nums, signals,dims]=lerconfigs(folder)

%folder como 'b31'
%numother e o numero de sinais usados para reconstruir o target

%%return values:
%
%nums is the order of each signal (others and aim) in val
%
%signals is a cell array of strings each one contains the name of one signal, the last is the aim
%
%dims contains on each line the layer dimensions for the autoencoder for each signal

nome=strcat('../',folder,'/config.txt');

%disp(nome);

fid = fopen (nome, 'r');

numother=fscanf (fid,'%d',1);

others=zeros(1,numother);

others=fscanf (fid,'%d',numother);

target=fscanf (fid,'%d',1);

nums=[numother,others(:)',target];

signals=cell(numother+1,1);

dims=zeros(numother+1,3);



for i=1:numother+1

 signals{i,1}=fscanf (fid,'%s',1);

 dims(i,:)=fscanf (fid,'%d',3);
  

end