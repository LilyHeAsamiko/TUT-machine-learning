function [mu,sigma,covariance,p,index,N]=cifar_10_bayes_learn(F,labels)
%F=tr(1:1000,:);
%labels=tr_labels(1:1000);
M=length(labels);
% for i=1:10
%     index(i,:)=find(labels==(i-1))';
% end
index=cell(10,1);
mu=zeros(10,3);
sigma=zeros(10,3);
covariance=zeros(10,3,3);
for i=1:10
    index{i}=find(labels==(i-1));
    N(i)=length(index{i});
    mu(i,:)=mean(F(index{i},:),1);
    sigma(i,:)=std(F(index{i},:),1);
    covariance(i,:,:)=cov(F(index{i},:));
    p(i)=N(i)/M;
end
end