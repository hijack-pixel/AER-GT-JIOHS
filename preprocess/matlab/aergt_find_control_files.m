function file_names = aergt_find_control_files(folder_path, control_name, mouse_idx)
%AERGT_FIND_CONTROL_FILES Find TIFF chunks for one control mouse.

all_files = dir(fullfile(folder_path, '*.tif'));
if isempty(all_files)
    error('aergt_find_control_files:NoTiffFiles', ...
        'No .tif files found in %s.', folder_path);
end

all_names = {all_files.name};

switch lower(control_name)
    case 'con1'
        pattern = sprintf('^CON %d(?!\\d).*\\.tif$', mouse_idx);
        match_idx = ~cellfun(@isempty, regexp(all_names, pattern, 'once'));
        file_names = all_names(match_idx);
        file_names = sort(file_names);

    case 'con2'
        pattern = sprintf('^CON\\s+%d(\\.tif|@\\d+\\.tif)$', mouse_idx);
        match_idx = ~cellfun(@isempty, regexp(all_names, pattern, 'once'));
        file_names = all_names(match_idx);
        chunk_idx = zeros(numel(file_names), 1);
        for idx = 1:numel(file_names)
            token = regexp(file_names{idx}, sprintf('^CON\\s+%d@(\\d+)\\.tif$', mouse_idx), 'tokens', 'once');
            if isempty(token)
                chunk_idx(idx) = 0;
            else
                chunk_idx(idx) = str2double(token{1});
            end
        end
        [~, order_idx] = sort(chunk_idx, 'ascend');
        file_names = file_names(order_idx);

    otherwise
        error('aergt_find_control_files:UnknownControl', ...
            'Unknown control set: %s.', control_name);
end
end
