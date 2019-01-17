function trlabelst=cifar_10_1NN(x,trdata,trlabel)
[mtr,ntr]=size(trdata);
[mte,nte]=size(x);
for i=1:mte
    for j=1:mtr
        T=(trdata(j,:)==x(i,:));
        a(j)=sum(T(:));
    end
    [M,k]=max(a); 
    trlabelst(i)=trlabel(k);
end
end

