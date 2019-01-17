function net=cifar_10_MLP_train(tr,trlabels)
tr=double(tr);
trlabels=double(trlabels);
M=length(tr);
N=10;
target=zeros(M,N);
for i = 1:M
	for j=1:N
        target(i,trlabels(i)+1)=1;
    end
end

hsize=10;
net=patternnet(hsize);
net.layers{1}.transferFcn='logsig';
net=train(net,tr',target');
view(net);

end