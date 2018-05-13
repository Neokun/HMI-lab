%clear all, close all,

%my_percent = 0.2;%sparsity percentage

%C_data is W, already known
%fid = fopen('061.dat');
%C_data = fscanf(fid,'%f',[57 Inf]);
%fclose(fid);
%C_data = C_data';

%Init label = motion matrix size 770 * 12 F * N * 3
%Label = [ones(size(C_data,1),1),zeros(size(C_data,1),1)];
%Label = repmat(Label,1,6);

%Init C_sparse = zeros size 57*12 
C_sparse = zeros(size(C_data,2),size(Label,2));

select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
ProjG = @(X,k)(abs(X) >= select(sort(abs(X), 'descend'),k));
%ProjX = @(X,k)X .* (abs(X) >= select(sort(abs(X), 'descend'),k));

groupN = size(C_sparse,2)/3; % along column dimension, group 3D coordinates
ksparse = ceil(size(C_sparse,1)*my_percent); % along row dimension, sparse components, and set threshold is 0.1
niter = 100; D = C_data; Yval=Label; 
% Regarding motion matrix, Yval denostes motion matrix; D denotes W; Xc denotes component matrix
gamma = 1.6/norm(D)^2;
Xc = C_sparse;
        for it=1:niter
            R = D*Xc-Yval;
            tmpXc = Xc - gamma * D'*R;
            groupXc = tmpXc.*tmpXc;
            gXc = [];
            for grp=1:groupN
                gXc = [gXc,sqrt( sum( groupXc(:,(grp-1)*3+1:grp*3), 2) ) ];
            end
            gXc = ProjG(gXc,ksparse);
            groupXc = groupXc*0;
            for grp=1:groupN
                groupXc(:,(grp-1)*3+1:grp*3) = repmat(gXc(:,grp),1,3);
            end
            Xc = tmpXc.*groupXc;
%            Xc = ProjX(Xc - gamma * D'*R, ksparse);
        end
C_sparse = Xc;        
err = norm((Label-C_data*C_sparse)'*(Label-C_data*C_sparse));

% transition = 50; % varying
% Id = sparse([eye(size(C_data,1));zeros(size(C_data,1))]);
% Id = [Id,sparse(zeros(2*size(C_data,1),transition)),sparse([zeros(size(C_data,1));eye(size(C_data,1))])];
% C_data = [C_data;zeros(transition,size(C_data,2));C_data];
% Label = [ones(size(C_data,1),1),zeros(size(C_data,1),1)];
% 
% err = norm(Id*C_data-[D;D]);

