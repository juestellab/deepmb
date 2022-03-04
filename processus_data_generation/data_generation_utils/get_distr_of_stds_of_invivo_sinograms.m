function pd = get_distr_of_stds_of_invivo_sinograms(path_to_sinograms)
D_sinos = dir(fullfile(path_to_sinograms, '*.nii'));

max_vals = zeros(length(D_sinos),1);

for i_sino = 1: length(D_sinos)

    [~, name, ext] = fileparts(D_sinos(i_sino).name);
    sino = niftiread(fullfile(path_to_sinograms, [name ext]));
    max_vals(i_sino) = std(sino(:));
end

pd = fitdist(max_vals, 'kernel');

end
