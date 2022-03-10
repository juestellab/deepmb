function [rec_img] = reconstruct_model_based(model, sinogram, regularization, lambda_shearlet, lambda_tikhonov, lambda_laplacian, num_iterations_mb)
% Perform model-based reconstruction

if strcmp(regularization, 'L1_SHEARLET')
  rec_img = rec_nn_with_Shearlet_reg(model, sinogram, num_iterations_mb, lambda_shearlet);  
  
elseif strcmp(regularization, 'L2_TIKHONOV_AND_LAPLACIAN')
  RegL2 = @(x) x;
  RegL2T = @(x) x;  
  RegL2_lap = @(x) laplacian_per_wavelength(reshape(x, model.Discretization.sizeOfPixelGrid(2), model.Discretization.sizeOfPixelGrid(1), []));
  RegL2T_lap = @(x) laplacian_per_wavelength(reshape(x, model.Discretization.sizeOfPixelGrid(2), model.Discretization.sizeOfPixelGrid(1), []));
  
  rec_img = rec_nn_with_L2_reg(model, sinogram, num_iterations_mb, lambda_tikhonov, RegL2, RegL2T, lambda_laplacian, RegL2_lap, RegL2T_lap);
  
else
  disp(['Unknown regularisation: ' regularization]);
end

rec_img = fliplr(rec_img);
end
