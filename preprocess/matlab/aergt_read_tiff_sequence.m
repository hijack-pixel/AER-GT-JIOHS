function video = aergt_read_tiff_sequence(folder_path, file_list)
%AERGT_READ_TIFF_SEQUENCE Read and concatenate a list of TIFF stacks.

if isempty(file_list)
    error('aergt_read_tiff_sequence:EmptyList', 'No TIFF files were provided.');
end

total_frames = 0;
height = [];
width = [];
for idx = 1:numel(file_list)
    info = imfinfo(fullfile(folder_path, file_list{idx}));
    if isempty(height)
        height = info(1).Height;
        width = info(1).Width;
    elseif info(1).Height ~= height || info(1).Width ~= width
        error('aergt_read_tiff_sequence:SizeMismatch', ...
            'TIFF size mismatch in %s.', file_list{idx});
    end
    total_frames = total_frames + numel(info);
end

video_uint = zeros(height, width, total_frames, 'uint16');
cursor = 1;
for idx = 1:numel(file_list)
    file_path = fullfile(folder_path, file_list{idx});
    info = imfinfo(file_path);
    for frame_idx = 1:numel(info)
        video_uint(:, :, cursor) = imread(file_path, frame_idx);
        cursor = cursor + 1;
    end
end

video = double(video_uint);
end
