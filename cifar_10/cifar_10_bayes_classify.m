function c1=cifar_10_bayes_classify(f,mu,sigma,covariance,p)
%f=te;
%covariance=c................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................ovariance;
for i=1:10    
    h1(:,i)=normpdf(f(:,1),mu(i,1),sigma(i,1)).*normpdf(f(:,2),mu(i,2),sigma(i,2)).*normpdf(f(:,3),mu(i,3),sigma(i,3)).*p(i);    
%     h2(:,i)=mvnpdf(f,mu(i,:),abs([covariance(i,:,1);covariance(i,:,2);covariance(i,:,3)]));
end
[~,c1]=max(h1,[],2);
%[~,c2]=max(h2,[],2);
c1 = c1-1;
%c2 = c2-1;
end