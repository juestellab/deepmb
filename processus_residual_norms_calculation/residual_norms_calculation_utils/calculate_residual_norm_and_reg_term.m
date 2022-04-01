% [INPUT]
% path_and_filename_sinogram: Full path+filename to the target sinogram
% path_and_filename_recon:    Full path+filename to the target reconstruction
% model:                      Model to calculate the forward pass
% sinogram_mask:              Model-dependant binary mask for the sinograms
% is_bp:                      Boolean to indicate if this reconstruction is back-propagation
% calculate_reg_term:         Boolean to indicate if the regularization term must be calculated as well
% shearlet_l1_functional:     Apparatus to compute the model-based shearlet l1 functional
% calculate_residual_norm:    Boolean to indicate if the calculation process must be carried out (it can be skipped for MB and/or BP to save computation time)
% [OUTPUT]
% data_residual_norm:         Data residual norm
% reg_term:                   Regularization term
% sqared_2_norm_of_sinogram:  Squared 2-norm of the sinogram


function [data_residual_norm, reg_term, sqared_2_norm_of_sinogram] = calculate_residual_norm_and_reg_term(...
  path_and_filename_sinogram,...
  path_and_filename_recon,...
  model,...
  sinogram_mask,...
  is_bp,...
  calculate_reg_term,...
  shearlet_l1_functional,...
  calculate_residual_norm)

if calculate_residual_norm
  
  % Read the NIFTI files corresponding to the raw signal and reconstructed pressure field
  sinogram = double(niftiread(path_and_filename_sinogram).*sinogram_mask);
  recon = double(niftiread(path_and_filename_recon));
  
  % Threshold negative values, if needed
  if is_bp
    recon = max(0, recon);
  end
  
  % Apply the forward model pass to transform the reconstructed pressure field into an estimated sinogram
  sin_from_recon = model.Funcs.applyForward(recon).*sinogram_mask;
  
  % Apply a normalization to the estimated sinogram
  sin_from_recon = sin_from_recon .* sum(sin_from_recon(:).*sinogram(:)) ./ sum(sin_from_recon(:).^2);
  
  % Calculate the norm 2 squared
  sqared_2_norm_of_sinogram_estimation_minus_reference = norm(sin_from_recon(:)-sinogram(:), 2)^2;
  sqared_2_norm_of_sinogram = norm(sinogram(:), 2)^2;
  
  % Calculate the data residual norm
  data_residual_norm = sqared_2_norm_of_sinogram_estimation_minus_reference / sqared_2_norm_of_sinogram;
  
  % Calculate the regularization term
  if calculate_reg_term
    reg_term = shearlet_l1_functional(recon) / sqared_2_norm_of_sinogram;
  else
    reg_term = 0;
  end
  
  % In some cases, calculating the data residual norm and regualrization term for MB and BP is not useful as it can be obtained from a previous calculation...
  % ... therefore skipping this operation can save computation time
else
  data_residual_norm = 0;
  reg_term = 0;
  sqared_2_norm_of_sinogram = 0;
end

end