

function randlabel=cifar_10_rand(x)
[m,n]=size(x);
randlabel=randi([0,9],m,1);
end