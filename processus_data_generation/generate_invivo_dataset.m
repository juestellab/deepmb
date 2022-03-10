% Script to generate an in-vivo test dataset for DeepMB.
[...
    output_root_folder,...
    path_to_rec_toolbox,...
    device_id,...
    speed_of_sound_range,...
    N,...
    field_of_view,...
    use_eir,...
    use_sir,...
    use_single_speed_of_sound,...
    filt_min,...
    filt_max,...
    num_cropped_samples_at_sinogram_start,...
    use_indiv_eir,...
    regularization,...
    lambda_shearlet,...
    lambda_tikhonov,...
    lambda_laplacian,...
    num_iterations_mb,...
    image_folder,...
    splits,...
    voc2012_start_indices_per_split,...
    num_considered_voc2012_imgs_per_split,...
    number_of_different_gt_sos_for_each_img_per_split,...
    use_gt_sos_also_as_rec_sos,...
    number_of_different_rec_sos_for_each_img_per_split,...
    random_sinogram_scaling,...
    max_scale_factor_for_initial_images,...
    path_to_invivo_sinograms_for_std_distr_estimation,...
    initial_img_transform,...
    path_to_studies_folders_with_raw_data,...
    data_name_prefix,...
    path_to_csv,...
    func_name_pulse_suffix,...
    broken_transducers,...
    use_all_sos_instead_of_only_gt_sos_for_invivo_rec] = set_parameters_for_data_generation();

%% Initialize reconstructions
run([path_to_rec_toolbox filesep 'startup_reconstruction.m']);
addpath(genpath('data_generation_utils'));
K = load_or_calculate_kernel_for_backprojection_rec(deviceId);

%% Define output properties
split = 'test';
subfolder_sinograms = 'sinograms';
subfolder_recs = 'rec_images';
subfolder_sos = 'sos_sim_and_rec';
subfolder_bp = 'backprojection';
create_folder_structure_of_dataset(output_root_folder, {split}, {subfolder_sinograms, subfolder_recs, subfolder_sos, subfolder_bp});
copyfile('./set_parameters_for_data_generation.m', [output_root_folder filesep datestr(now, 'yy-mm-dd_') 'parameters_for_data_generation.m']);

%% Main Loop: Load sinogram, reconstruct
[studies, scans, selmats, speed_of_sounds_invivo_gt, ~, names] = read_studies_scans_selmats_from_csv(path_to_csv, 1);

for speed_of_sound = speed_of_sound_range
    fprintf('Loop 1: Generate model for speed of sound "%i".\n', speed_of_sound);
    model = define_model_for_reconstruction(...
        field_of_view, N, device_id, use_eir, use_indiv_eir, use_sir, use_single_speed_of_sound, speed_of_sound,...
        num_cropped_samples_at_sinogram_start, filt_min, filt_max);
    if use_all_sos_instead_of_only_gt_sos_for_invivo_rec
        selmats_with_current_rec_sos = 1:length(selmats);
    else
        selmats_with_current_rec_sos = find(speed_of_sounds_invivo_gt==speed_of_sound);
        selmats_with_current_rec_sos = selmats_with_current_rec_sos';
    end
    
    for i_frame_with_current_rec_sos = selmats_with_current_rec_sos
        % Load image
        study = studies(i_frame_with_current_rec_sos);
        scan = scans(i_frame_with_current_rec_sos);
        selmat = selmats{i_frame_with_current_rec_sos};
        sos_gt = speed_of_sounds_invivo_gt(i_frame_with_current_rec_sos);
        name_part_1 = [data_name_prefix names{i_frame_with_current_rec_sos} '_sosGt_' num2str(sos_gt) '_sosRec_' num2str(speed_of_sound)];
        
        % Load and preprocess sinogram
        path_msot_file = [path_to_studies_folders_with_raw_data filesep 'Study_' num2str(study) filesep 'Scan_' num2str(scan) filesep 'Scan_' num2str(scan) '.msot'];
        data_raw = loadMSOTSignals(path_msot_file, selmat);

        data = crop_first_n_signals(data_raw,  1);
        data = apply_butterworth_window_to_sinogram(data, 2, 300, size(data,1)-200);
        data = filter_butter_zero_phase(data, model.Probe.DAC.frequency, [model.DataPreprocessing.filtCutoffMin, model.DataPreprocessing.filtCutoffMax],true);
        data = crop_first_n_signals(data, model.DataPreprocessing.numCroppedSamplesAtSinogramStart - 1);
        data_preprocessed = interpolate_signals_of_broken_transducers(data, broken_transducers);

        for wavelength = 1:size(data_preprocessed,3)
            name = [name_part_1 func_name_pulse_suffix(wavelength)];
            niftiwrite(single(data_preprocessed(:,:,wavelength)), fullfile(output_root_folder, split, subfolder_sinograms, [name '.nii']));
        end

        % Save gt and rec speed of sound in csv file
        for wavelength = 1:size(data_preprocessed, 3)
            name = [name_part_1 func_name_pulse_suffix(wavelength)];
            writematrix([sos_gt; speed_of_sound], fullfile(output_root_folder, split, subfolder_sos, [name '.csv']));
        end
        
        % Apply and save backprojection reconstruction
        rec_img = reconstruct_bp(model, K, data_preprocessed);
        for wavelength = 1:size(data_preprocessed,3)
            name = [name_part_1 func_name_pulse_suffix(wavelength)];
            niftiwrite(single(rec_img(:,:,wavelength)), fullfile(output_root_folder, split, subfolder_bp, [name '.nii']));
        end

        % Apply and save model-based reconstruction
        rec_img = reconstruct_model_based(model, data_preprocessed, regularization, lambda_shearlet, lambda_tikhonov, lambda_laplacian, num_iterations_mb);
        for wavelength = 1:size(data_preprocessed,3)
            name = [name_part_1 func_name_pulse_suffix(wavelength)];
            niftiwrite(single(rec_img(:,:,wavelength)), fullfile(output_root_folder, split, subfolder_recs, [name '.nii']));
        end
    end
end
