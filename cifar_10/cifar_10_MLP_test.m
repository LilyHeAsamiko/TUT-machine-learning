function classes=cifar_10_MLP_test(te,net);
te=double(te);
y = net(te');
classes = vec2ind(y)-1;