function create_folder_structure_of_dataset(root_path, subfolder_names, subsubfolder_names)

if ~exist(root_path, 'dir')
   mkdir(root_path);
end

for subfolder_name_as_cell = subfolder_names
    subfolder_name = subfolder_name_as_cell{1};
    
    subfolder_path = fullfile(root_path, subfolder_name);
    if ~exist(subfolder_path, 'dir')
        mkdir(subfolder_path);
    end
    
    for subsubfolder_name_as_cell = subsubfolder_names
        subsubfolder_name = subsubfolder_name_as_cell{1};
        
        subsubfolder_path = fullfile(subfolder_path, subsubfolder_name);
        if ~exist(subsubfolder_path, 'dir')
            mkdir(subsubfolder_path);
        end
    end
end

