function [model] = define_msot_model(device_id, speed_of_sound, model_normalization_factor)
% Define the MSOT model for DeepMB, with FOV (m), image dimensions (px), and filter frequencies

if nargin < 3
    model_normalization_factor = [];
end

N = [416 416];                                     % Image dimension (px)
fieldOfView = [-0.02075 0.02075 -0.02075 0.02075]; % Field of view (m)
use_eir = true;                                    % Electrical impulse response
use_sir = true;                                    % Spatial impulse response
use_single_speed_of_sound = true;                  % Indicate whether the probe couplant should be defined by a specific SoS
filt_min = 1e5;                                    % Lower bound of the bandpass filter (Hz)
filt_max = 12e6;                                   % Higher bound of the bandpass filter (Hz)
num_cropped_samples_at_sinogram_start = 110;       % Number of samples that are cropped from the beginning of the sinogram (in signal samples)
use_indiv_eir = false;                             % Indicate whether the EIR of each individual element should be taken into account

model = define_model_for_reconstruction(...
  fieldOfView, N, device_id, use_eir, use_indiv_eir, use_sir, use_single_speed_of_sound, speed_of_sound,...
   num_cropped_samples_at_sinogram_start, filt_min, filt_max, model_normalization_factor);
end

