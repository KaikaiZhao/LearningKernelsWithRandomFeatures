function Z = createArccosineFeatures( D, W, X )
%CREATEARCCOSINEFEATURES Summary of this function goes here
%   Detailed explanation goes here
% creates Arccosine random features
% Inputs:
% D the number of features to make
% W the parameters for those features (d x D and 1 x D)
% X the datapoints to use to generate those features (d x N)
wTx = X*W; % 5000*20000
H_wTx_temp = sign(wTx);
H_wTx = 0.5 * (1 + H_wTx_temp); %5000*20000
Z = sqrt(2/D)*H_wTx.*(wTx.^2); %5000*20000 sqrt(2/D)*
end

