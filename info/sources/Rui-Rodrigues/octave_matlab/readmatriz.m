function A=readmatriz(filename)


fid = fopen (filename, 'r');

nlin=0;

ncol=0;

B=fscanf (fid,'%u',2);

nlin=B(1);
ncol=B(2);

%disp(nlin);
%disp(ncol);

aux=zeros(nlin,ncol);

aux=fscanf (fid,'%f',nlin*ncol);

% for i=1:nlin

%   for j=1:ncol

%     A(i,j)=fscanf (fid,'%f',1);

%   endfor

% endfor



aux=reshape(aux,ncol,nlin);

A=aux';

end