% Script to synthesize a training and validation dataset for DeepMB.
[...
    output_root_folder,...
    path_to_rec_toolbox,...
    deviceId,...
    speed_of_sound_range,...
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
    acuity_data_studies_path,...
    data_name_prefix,...
    path_to_csv,...
    func_name_pulse_suffix,...
    broken_transducers,...
    use_all_sos_instead_of_only_gt_sos_for_invivo_rec] = set_parameters_for_data_generation();

%% Initialize reconstructions
run([path_to_rec_toolbox filesep 'startup_reconstruction.m']);
addpath(genpath('data_generation_utils'));
K = load_or_calculate_kernel_for_backprojection_rec(deviceId);

%% Create output folder strucutre and save parameter file
subfolder_sinograms = 'sinograms';
subfolder_sos = 'sos_sim_and_rec';
subfolder_recs = 'rec_images';
subfolder_initial = 'initial_images';
subfolder_bp = 'backprojection';

create_folder_structure_of_dataset(output_root_folder, splits, {subfolder_sinograms, subfolder_sos, subfolder_recs, subfolder_initial, subfolder_bp});
copyfile('./set_parameters_for_data_generation.m', [output_root_folder filesep 'parameters_for_data_generation.m']);

%% Part 1: Calculate split ids, image indices, sos_gt, sos_rec, split for all generated data
split_ids_of_generated_data = [];
voc2012_image_indices = [];
speed_of_sounds_gt = [];
speed_of_sounds_rec = [];

randStream = RandStream('mlfg6331_64', 'Seed', 1);
for split_id = 1: length(splits)
    total_recs_per_split = num_considered_voc2012_imgs_per_split(split_id)*number_of_different_gt_sos_for_each_img_per_split(split_id)*number_of_different_rec_sos_for_each_img_per_split(split_id);
    split_ids_of_generated_data = vertcat(split_ids_of_generated_data, repmat(split_id, total_recs_per_split, 1)); %#ok<AGROW>
    
    for i_outer = 1: num_considered_voc2012_imgs_per_split(split_id)
        recs_per_image = number_of_different_gt_sos_for_each_img_per_split(split_id)*number_of_different_rec_sos_for_each_img_per_split(split_id);
        voc2012_image_indices = vertcat(voc2012_image_indices, repmat(voc2012_start_indices_per_split(split_id)+i_outer-1, recs_per_image, 1)); %#ok<AGROW>
        
        speed_of_sounds_gt_per_img = datasample(randStream, speed_of_sound_range', number_of_different_gt_sos_for_each_img_per_split(split_id), 'Replace', false);
        
        for j = 1 : number_of_different_gt_sos_for_each_img_per_split(split_id)
            speed_of_sounds_gt = vertcat(speed_of_sounds_gt, repmat(speed_of_sounds_gt_per_img(j), number_of_different_rec_sos_for_each_img_per_split(split_id), 1)); %#ok<AGROW>
            speed_of_sounds_rec = vertcat(speed_of_sounds_rec, datasample(randStream, speed_of_sound_range', number_of_different_rec_sos_for_each_img_per_split(split_id), 'Replace', false)); %#ok<AGROW>
        end
    end
end

if use_gt_sos_also_as_rec_sos
    fprintf('SoS_gt is also used as SoS_rec.\n');
    speed_of_sounds_rec = speed_of_sounds_gt;
end

%% Part 2: Generate and save simulated sinograms, model-based images, and backprojection images.
D_images = dir([image_folder '/*.jpg']);
rng(234);
D_images = D_images(randperm(length(D_images)));

rng(2609);
if strcmpi(random_sinogram_scaling, 'UNIFORM_RANGE')
    scaling_factors_for_voc2012_images = rand(1, length(D_images)) * max_scale_factor_for_initial_images;
    
elseif strcmpi(random_sinogram_scaling, 'MATCH_STD_DISTR_OF_INVIVO_SINOGRAMS')
        pd_of_test_sinogram_stddevs = get_distr_of_stds_of_invivo_sinograms(path_to_invivo_sinograms_for_std_distr_estimation);
        std_devs_for_sinograms_generated_from_VOC2012 = random(pd_of_test_sinogram_stddevs, 1, length(D_images));
        
else
    error(['Unknown parameter for random sinogram scaling: "' random_sinogram_scaling '".\n']);
end
batch_size = 8*32; % Reconstruct multiple images simultaneously to make use of SIMD instructions.

for speedOfSound = speed_of_sound_range
    fprintf('Generate model for speed of sound "%i".\n', speedOfSound);
    [model] = define_msot_model(deviceId, speedOfSound);
    samples_with_current_sos = find(speed_of_sounds_gt==speedOfSound);
    
    for i_sample_with_current_sos = 1 : batch_size : length(samples_with_current_sos)
        samples_in_current_batch = samples_with_current_sos(i_sample_with_current_sos : min(i_sample_with_current_sos+batch_size-1, length(samples_with_current_sos)));
        
        % Load and prepare ground truth initial pressure images
        img_batch = zeros(model.Discretization.sizeOfPixelGrid(1), model.Discretization.sizeOfPixelGrid(2), length(samples_in_current_batch));
        name_batch = {};
        for i_batch = 1:length(samples_in_current_batch)
            current_img = voc2012_image_indices(samples_in_current_batch(i_batch));
            [~, name, ext] = fileparts(D_images(current_img).name);
            name_batch{i_batch} = [name '_sosGt_' num2str(speed_of_sounds_gt(samples_in_current_batch(i_batch))) '_sosRec_' num2str(speed_of_sounds_rec(samples_in_current_batch(i_batch)))]; %#ok<SAGROW>
            % Load img, transform to gray, resize, scale
            img = double(imread(fullfile(D_images(current_img).folder, D_images(current_img).name))); 
            img_gray = 0.2989 * img(:,:,1);
            if(size(img,3)>1)
                img_gray = img_gray + 0.5870 * img(:,:,2) + 0.1140 * img(:,:,3);
            end
            img_gray = imresize(img_gray, model.Discretization.sizeOfPixelGrid, 'bicubic');
            img_gray = mat2gray(img_gray);
            
            if strcmpi(random_sinogram_scaling, 'UNIFORM_RANGE')
                img_gray = img_gray * scaling_factors_for_voc2012_images(voc2012_image_indices(samples_in_current_batch(i_batch)));
            end
            
            % Apply transform for initial image if specified
            if isa(initial_img_transform, 'function_handle')
                img_gray = initial_img_transform(img_gray);
            end
                        
            img_batch(:,:,i_batch) = img_gray;
        end

        % Save gt and rec speed of sound in csv file
        for i_batch = 1:length(samples_in_current_batch)
            sos_gt = speed_of_sounds_gt(samples_in_current_batch(i_batch));
            sos_rec = speed_of_sounds_rec(samples_in_current_batch(i_batch));
            split = splits{split_ids_of_generated_data(samples_in_current_batch(i_batch))};
            writematrix([sos_gt; sos_rec], fullfile(output_root_folder, split, subfolder_sos, [name_batch{i_batch} '.csv']));
        end

        % Apply forward model and save the resulting sinogram
        sinograms = model.Funcs.applyForward(img_batch);
        sinograms = reshape(sinograms,[] , model.Probe.detector.numOfTransducers, size(img_batch,3));
        
        for i_batch = 1:length(samples_in_current_batch)
            if strcmpi(random_sinogram_scaling, 'MATCH_STD_DISTR_OF_INVIVO_SINOGRAMS')
                % Scale sinograms so that their stds match the provided distribution of test stds
                current_std = std(vec(sinograms(:,:,i_batch)));
                current_scaling = std_devs_for_sinograms_generated_from_VOC2012(voc2012_image_indices(samples_in_current_batch(i_batch))) / current_std;
                sinograms(:,:,i_batch) = sinograms(:,:,i_batch) * current_scaling;

                % Apply scaling also to initial image
                img_batch(:,:,i_batch) = img_batch(:,:,i_batch) * current_scaling;
            end
            
            split = splits{split_ids_of_generated_data(samples_in_current_batch(i_batch))};
            niftiwrite(single(sinograms(:,:,i_batch)), fullfile(output_root_folder, split, subfolder_sinograms, [name_batch{i_batch} '.nii']));
        end
        
        % Save ground truth initial pressure images
        for i_batch = 1:length(samples_in_current_batch)
            split = splits{split_ids_of_generated_data(samples_in_current_batch(i_batch))};
            niftiwrite(single(img_batch(:,:,i_batch)), fullfile(output_root_folder, split, subfolder_initial, [name_batch{i_batch} '.nii']));
        end
        
        % Apply and save backprojection reconstruction for the split 'val'
        rec_imgs_bp = reconstruct_bp(model, K, sinograms);
        for i_batch = 1:length(samples_in_current_batch)
            split = splits{split_ids_of_generated_data(samples_in_current_batch(i_batch))};
            if strcmpi(split, 'val')
                niftiwrite(single(rec_imgs_bp(:,:,i_batch)), fullfile(output_root_folder, split, subfolder_bp, [name_batch{i_batch} '.nii']));
            end
        end
        
        % Apply and save model-based reconstruction
        rec_imgs = reconstruct_model_based(model, sinograms);
        for i_batch = 1:length(samples_in_current_batch)
            split = splits{split_ids_of_generated_data(samples_in_current_batch(i_batch))};
            niftiwrite(single(rec_imgs(:,:,i_batch)), fullfile(output_root_folder, split, subfolder_recs, [name_batch{i_batch} '.nii']));
        end
    end
end
