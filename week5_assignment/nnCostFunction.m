function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% forward propagation
X = [ones(1, m); X'];
a1 = X;  
z2 = Theta1 * a1;
a2 = [ones(1, m); sigmoid(z2)]; 
z3 = Theta2 * a2;
a3 = sigmoid(z3);  

% Convert y into a matrix of labels
y_matrix = zeros(num_labels, m);
y_matrix(sub2ind(size(y_matrix), y', 1:m)) = 1;

% Cost function without regularization
J = (1/m) * sum(sum(-y_matrix .* log(a3) - (1 - y_matrix) .* log(1 - a3)));

% Add regularization for layer one
J = J + (lambda / (2*m)) * sum(sum(Theta1(:, 2:end) .^ 2));

% Add regularization for layer two
J = J + (lambda / (2*m)) * sum(sum(Theta2(:, 2:end) .^ 2));

% error of output layer
D3 = a3 - y_matrix;
D2 = (Theta2' * D3) .* [ones(1, m); sigmoidGradient(z2)];

% calculating the gradient of D
Theta2_grad = (1/m) * D3 * a2';
Theta1_grad = (1/m) * D2(2:end, :) * a1';

% Add gradient regularization.
Theta2_grad = Theta2_grad + (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);

grad = [Theta1_grad(:); Theta2_grad(:)];


end
