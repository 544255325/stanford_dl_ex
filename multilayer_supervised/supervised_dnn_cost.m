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

%% compute cost
%%% YOUR CODE HERE %%%

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



