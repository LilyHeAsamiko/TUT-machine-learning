function C2=cifar_10_bayes_classifyext(f,mu,sigma,covariance,p,Size)
%f=Te;
%mu=Mu;
%sigma=Sigma;
%p=P;
%covaraince=Covariance;
[M,L]=size(f);
% Covariance(:,:,:)=zeros(M,L,L);
for i=1:10    
% %    h(:,i)=exp(-(f(:,1)-mu(i,1)).^2/(2*sigma(i,1).^2))./(2*pi*sigma(i,1).^2).*exp(-(f(:,2)-mu(i,2)).^2./(2*sigma(i,2).^2))./(2*pi*sigma(i,2).^2).*(f(:,3)-mu(i,3)).^2./(2*pi*sigma(i,3).^2).*p(i);
% % h1=ones(M,L,L);
% %     for j=1:L
% %         h1(i,:,:)=h1(i,j,:).*normpdf(f(:,j),mu(i,j),sigma(i,j)); 
% %     end
% %     h1(i,:,:)=h1(i,:,:).*p(i);
%     for l=1:L-1
%         Covariance(i,l:l+1,:)=[Covariance(i,l,:);Covariance(i,l+1,:)];
% %     for j=1:L-1
% %     Covariance{i,l,j}=[Covariance{i,l,j};Covariance{i,l,j+1}];
% %     end
% %     Covariance{i,l,:}=[Covariance{1,l,:};Covariance{i,l+1,:}];
%     end
% % h1(:,i)=h1(:,:).*p(i);    
% %end
A(:,:)=covariance(i,:,:);
% Covariance_(:,:)=(A'*A)
h2(:,i)=mvnpdf(f,mu(i,:),A(:,:))*p(i);
end
% [~,C1]=max(h1,[],2);
[~,C2]=max(h2,[],2);
% C1 =C1-1;
C2 =C2-1;
end