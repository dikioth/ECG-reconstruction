function A=writematriz(filename,matriz)


fid = fopen (filename, 'w');

[nrows, ncols]=size(matriz);

fprintf(fid,'%d ',nrows);

fprintf(fid,'%d\n',ncols);

for i=1:nrows
  fprintf(fid,'%10.7f ',matriz(i,:));
  fprintf(fid,'\n');
end	  

fclose (fid);

end