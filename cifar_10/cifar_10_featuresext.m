function F=cifar_10_features(X,Size)
%X=tr_data(i,:);
%Size=16;
Xr=reshape(X(1:1024),[32 32]);
Xg=reshape(X(1025:2048),[32 32]);
Xb=reshape(X(2049:3072),[32 32]);
% for i=1:(32/Size)^2
% Mr(i)=mean(mean(X(i:i+Size^2-1)));
% Mg(i)=mean(mean(X(1024+i:1024+i+Size^2-1)));
% Mb(i)=mean(mean(X(2048+i:2048+i+Size^2-1)));
% end
for i=1:(32/Size)
    Mr(i)=mean(mean(X(i:i+Size-1)));
    Mg(i)=mean(mean(X(1024+i:1024+i+Size-1)));
    Mb(i)=mean(mean(X(2048+i:2048+i+Size-1)));
end 
    F=[Mr,Mg,Mb];
end