
function accuracy=cifar_10_evaluate(pred,gt)
COUNT=sum(pred(:)==gt(:));
accuracy=COUNT/length(pred);
end
