function [rec_img] = reconstruct_model_based(model,sinogram)
% Perform model-based reconstruction with fixed parameters

num_iter_nn = 50;
REGULARIZATION = 'SHEARLET'; % Choices: ('SHEARLET', 'L2')

if strcmp(REGULARIZATION, 'SHEARLET')
  lambdaShearlet = 1e-2;
  rec_img = rec_nn_with_Shearlet_reg(model, sinogram, num_iter_nn, lambdaShearlet);  
  
elseif strcmp(REGULARIZATION, 'L2')
  lambdaL2 = 5e-3;
  RegL2 = @(x) x;
  RegL2T = @(x) x;  
  rec_img = rec_nn_with_L2_reg(model, sinogram, num_iter_nn, lambdaL2, RegL2, RegL2T, 0, [], []);
  
else
  disp(['Unknown regularisation: ' REGULARIZATION]);
end

rec_img = fliplr(rec_img);
end
