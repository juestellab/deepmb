% Calculate i) data residual norms and ii) regularization terms, from a given dataset split for all inferred DeepMB reconstructions,...
% ... together with the corresponding ground-truth model-based reconstructions and the traditional backprojection reconstructions.
% Residuals are saved at: "<path-to-inferred-nifti-recons>/../gt_and_inf_residuals".

clearvars;
close all;
clc;

%% Specify parameters
[...
  path_to_inferred_recons_nifti,...
  file_name_filter_inferrred_images,...
  path_to_dataset_split_with_gt_recs,...
  name_summary_file,...
  calculate_residual_norms_of_deepmb_recs,...
  calculate_residual_norms_of_mb_recs,...
  calculate_residual_norms_of_bp_recs,...
  calculate_reg_terms,...
  recalculate,...
  path_to_rec_toolbox,...
  use_model_sinogram_mask,...
  device_id,...
  N,...
  field_of_view,...
  use_eir,...
  use_sir,...
  use_single_speed_of_sound,...
  filt_min,...
  filt_max,...
  num_cropped_samples_at_sinogram_start,...
  use_indiv_eir] = set_parameters_for_residual_norms_calculation();


%% Set up environment and output folders
run(fullfile(path_to_rec_toolbox, 'startup_reconstruction.m'));
addpath('residual_norms_calculation_utils');
addpath('../processus_data_generation/data_generation_utils');

residual_norms_save_folder = fullfile(path_to_inferred_recons_nifti, '..', 'gt_and_inf_residual_norms');
if ~exist(residual_norms_save_folder, 'dir')
  mkdir(residual_norms_save_folder);
end


%% Main script body
% Find all pairs of inferred and gt recons
dir_inferred_recs = dir(fullfile(path_to_inferred_recons_nifti, file_name_filter_inferrred_images));
recs_for_residual_norms_calculation = {};
for i_inf_rec = 1:length(dir_inferred_recs)
  if exist(fullfile(path_to_dataset_split_with_gt_recs, 'rec_images', dir_inferred_recs(i_inf_rec).name), 'file') && exist(fullfile(path_to_dataset_split_with_gt_recs, 'backprojection', dir_inferred_recs(i_inf_rec).name), 'file')
    recs_for_residual_norms_calculation{end+1} = dir_inferred_recs(i_inf_rec).name; %#ok<SAGROW>
  end
end

% Load sos for all recons
soss_for_residual_norms_calculation = zeros(size(recs_for_residual_norms_calculation));
soss_gt = zeros(size(recs_for_residual_norms_calculation));
for i_rec_for_sos = 1:length(recs_for_residual_norms_calculation)
  path_csv_sos = fullfile(path_to_dataset_split_with_gt_recs, 'sos_sim_and_rec', strrep(recs_for_residual_norms_calculation{i_rec_for_sos}, '.nii', '.csv'));
  sos = csvread(path_csv_sos);
  soss_for_residual_norms_calculation(i_rec_for_sos) = sos(2);
  soss_gt(i_rec_for_sos) = sos(1);
end

% Iterate sos, build model and calculate residuals
sos_range = sort(unique(soss_for_residual_norms_calculation));
residual_norms = zeros(length(soss_for_residual_norms_calculation), 7);
wavelengths = zeros(length(soss_for_residual_norms_calculation), 1);
anatomies = cell(length(soss_for_residual_norms_calculation), 1);

for sos = sos_range
  model = define_model_for_reconstruction(...
    field_of_view, N, device_id, use_eir, use_indiv_eir, use_sir, use_single_speed_of_sound, sos,...
    num_cropped_samples_at_sinogram_start, filt_min, filt_max, 1.0);
  shearlet_l1_functional = get_shearlet_L1_functional(model.Discretization.sizeOfPixelGrid);
  if use_model_sinogram_mask
    sinogram_mask = get_sinogram_mask_of_model_reach(model);
  else
    sinogram_mask = ones(size(sinogram_mask));
  end
  ids_of_rec_with_current_sos = find(soss_for_residual_norms_calculation==sos);
  
  for id_rec = ids_of_rec_with_current_sos
    path_csv_with_residual_norms = fullfile(residual_norms_save_folder, strrep(recs_for_residual_norms_calculation{id_rec}, '.nii', '.csv'));
    
    % Get wavelength of current sample from file name
    pat_wavelength = '0nm';
    wavelength_char_start = strfind(recs_for_residual_norms_calculation{id_rec}, pat_wavelength);
    s = recs_for_residual_norms_calculation{id_rec};
    wavelength = str2double(s((wavelength_char_start-2) : (wavelength_char_start)));
    wavelengths(id_rec) = wavelength;
    
    % Get anatomy of current sample from file name
    anatomy_options = {'biceps','thyroid','carotid','calf','ulnar','neck','colon','breast'};
    anatomy_ids = cellfun(@(a) contains(recs_for_residual_norms_calculation{id_rec}, a), anatomy_options);
    if ~any(anatomy_ids)
      anatomy='unknown';
    else
      anatomy = anatomy_options{anatomy_ids};
    end
    anatomies{id_rec} = anatomy;
    
    % Load or calculate residual norms of current sample
    if ~recalculate && exist(path_csv_with_residual_norms, 'file')
      residual_norms(id_rec, :) = readmatrix(path_csv_with_residual_norms);
      
    else
      path_sinogram = fullfile(path_to_dataset_split_with_gt_recs, 'sinograms', recs_for_residual_norms_calculation{id_rec});
      path_gt_rec = fullfile(path_to_dataset_split_with_gt_recs, 'rec_images', recs_for_residual_norms_calculation{id_rec});
      path_bp_rec = fullfile(path_to_dataset_split_with_gt_recs, 'backprojection', recs_for_residual_norms_calculation{id_rec});
      path_inf_rec = fullfile(path_to_inferred_recons_nifti, recs_for_residual_norms_calculation{id_rec});
      
      [data_residual_norm_gt_rec, reg_term_gt_rec, sqared_2_norm_of_signal] = calculate_residual_norm_and_reg_term(path_sinogram, path_gt_rec, model, sinogram_mask, false, calculate_reg_terms, shearlet_l1_functional, calculate_residual_norms_of_mb_recs);
      [data_residual_norm_bp_rec, reg_term_bp_rec, ~] = calculate_residual_norm_and_reg_term(path_sinogram, path_bp_rec, model, sinogram_mask, true, calculate_reg_terms, shearlet_l1_functional, calculate_residual_norms_of_bp_recs);
      [data_residual_norm_inf_rec, reg_term_inf_rec, ~] = calculate_residual_norm_and_reg_term(path_sinogram, path_inf_rec, model, sinogram_mask, false, calculate_reg_terms, shearlet_l1_functional, calculate_residual_norms_of_deepmb_recs);
      
      residual_norms(id_rec, :) = [data_residual_norm_gt_rec, data_residual_norm_inf_rec, data_residual_norm_bp_rec, reg_term_gt_rec, reg_term_inf_rec, reg_term_bp_rec, sqared_2_norm_of_signal];
      writematrix(residual_norms(id_rec, :), path_csv_with_residual_norms);
    end
  end
end


%% Save all residuals to summary files
res_deepmb_rel_to_mb = residual_norms(:, 2) ./ residual_norms(:, 1);
res_bp_rel_to_mb = residual_norms(:, 3) ./ residual_norms(:, 1);

res_and_names = horzcat(...
  recs_for_residual_norms_calculation', ...
  num2cell(residual_norms),...
  num2cell(res_deepmb_rel_to_mb),...
  num2cell(res_bp_rel_to_mb),...
  num2cell(soss_for_residual_norms_calculation'),...
  num2cell(soss_gt'),...
  num2cell(wavelengths),...
  anatomies...
  );

column_names = {...
  'Name',...
  'Norm. data res. MB',...
  'Norm. data res. DeepMB',...
  'Norm. data res. BP',...
  'Norm. reg. res. MB',...
  'Norm. reg. res. DeepMB',...
  'sqared_2_norm_of_signal',...
  'Norm. reg res. BP',...
  'Rel. res. DeepMB/MB',...
  'Rel. res. BP/MB',...
  'Speed of sound rec',...
  'Speed of sound gt',...
  'Wavelength',...
  'Anatomy'
  };

res_and_names = vertcat(column_names, res_and_names);
writecell(res_and_names, fullfile(residual_norms_save_folder, [name_summary_file '.csv']));
writecell(res_and_names, fullfile(residual_norms_save_folder, [name_summary_file '.xlsx']));


%% Calculate average, standard deviation, and median
avg_mb = mean(residual_norms(:,1));
avg_deepmb = mean(residual_norms(:,2));
avg_bp = mean(residual_norms(:,3));

std_mb = std(residual_norms(:,1));
std_deepmb = std(residual_norms(:,2));
std_bp = std(residual_norms(:,3));

med_mb = median(residual_norms(:,1));
med_deepmb = median(residual_norms(:,2));
med_bp = median(residual_norms(:,3));

disp(['MB: Average = ' num2str(avg_mb) ', Std = ' num2str(std_mb) ', Median = ' num2str(med_mb)]);
disp(['DeepMB: Average = ' num2str(avg_deepmb) ', Std = ' num2str(std_deepmb) ', Median = ' num2str(med_deepmb)]);
disp(['BP: Average = ' num2str(avg_bp) ', Std = ' num2str(std_bp) ', Median = ' num2str(med_bp)]);


%% Violin plots
if license('test','statistics_toolbox')
  
  color_input_signal = [0.906,0.027,0.008];
  color_denoised_signal = [0.004,0.714,0.051];
  color_inferred_noise_signal = [0.071,0.216,0.612];
  color_4 = [0.906,0.616,0.008];
  
  mc = vertcat(color_inferred_noise_signal,color_inferred_noise_signal,color_inferred_noise_signal);
  medc = vertcat(color_input_signal,color_input_signal,color_input_signal);
  
  figure;
  violin(residual_norms(:,1:3),'xlabel',{'MB', 'DeepMB', 'BP'},'bw', [], 'support', [0 max(vec(residual_norms(:,1:3)))+0.01], 'mc', mc, 'medc', medc);
  ylim([0 max(vec(residual_norms(:,1:3)))+0.1]);
  xticks([1,2,3]);
  ylabel('Residual norm');
  title('Image fidelity of DeepMB, MB, and BP')
  savefig(fullfile(residual_norms_save_folder, [name_summary_file '_data_res_normalized.fig']));
end