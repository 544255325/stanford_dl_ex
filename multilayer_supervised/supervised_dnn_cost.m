function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
%隐藏层

for l = 1:numHidden
    if(l == 1)
        z = stack{l}.W * data;
    else
        z = stack{l}.W * hAct{l-1};
    end
    z = bsxfun(@plus,z,stack{l}.b);%%z:256*60000 b:256*1
    hAct{l} = 1./(1+exp(-z));
end
%输出层
h  = exp(bsxfun(@plus,stack{numHidden+1}.W * hAct{numHidden},stack{numHidden+1}.b));
pred_prob = bsxfun(@rdivide,h,sum(h,1));
hAct{numHidden+1} = pred_prob;%最后一层输出的实际上是预测的分类结果
  

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost计算softmax函数的损失函数
%%% YOUR CODE HERE %%%
logp = log2(pred_prob);
index = sub2ind(size(logp),labels',1:size(pred_prob,2));
ceCost = -sum(logp(index));

%% compute gradients using backpropagation

%%% YOUR CODE HERE %%%
%输出层
output = zeros(size(pred_prob));
output(index) = 1;
error = pred_prob - output;

for l = numHidden+1 : -1 :1
    gradStack{l}.b = sum(error,2);
    if(l == 1)
        gradStack{l}.W = error * data';
        break;
    else
        gradStack{l}.W = error * hAct{l-1}';
    end
    error = (stack{l}.W)'*error .* hAct{l-1} .* (1-hAct{l-1});
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
%penalty cost
wCost = 0;
for l = 1:numHidden+1
    wCost = wCost + 0.5*ei.lambda * sum(stack{l}.W(:) .^ 2);
end
cost = ceCost + wCost;

%gradient for non-bias terms
for l = numHidden:-1:1
    gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W;
end
 
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



