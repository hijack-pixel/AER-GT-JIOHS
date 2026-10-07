function video = aergt_read_tiff_stack(file_path, max_frames)
%AERGT_READ_TIFF_STACK Read a multi-frame TIFF stack as double.

if nargin < 2
    max_frames = [];
end

info = imfinfo(file_path);
num_frames = numel(info);
if ~isempty(max_frames)
    num_frames = min(num_frames, max_frames);
end

video_uint = zeros(info(1).Height, info(1).Width, num_frames, 'uint16');
for frame_idx = 1:num_frames
    video_uint(:, :, frame_idx) = imread(file_path, frame_idx);
end

video = double(video_uint);
end
