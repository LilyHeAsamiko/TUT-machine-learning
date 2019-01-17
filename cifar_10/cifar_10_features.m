function f=cifar_10_features(X)
mr=mean(mean(X(1:1024)));
mg=mean(mean(X(1025:2048)));
mb=mean(mean(X(2049:3072)));
f=[mr,mg,mb];
end