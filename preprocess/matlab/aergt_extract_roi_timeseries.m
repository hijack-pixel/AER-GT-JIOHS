function roi_traces = aergt_extract_roi_timeseries(video, roi_map, box_half)
%AERGT_EXTRACT_ROI_TIMESERIES Extract mean IOS traces from ROI boxes.

num_frames = size(video, 3);
num_rois = size(roi_map, 1);
roi_traces = zeros(num_frames, num_rois);

for roi_idx = 1:num_rois
    cx = roi_map(roi_idx, 1);
    cy = roi_map(roi_idx, 2);
    block = video(cy-box_half:cy+box_half, cx-box_half:cx+box_half, :);
    roi_traces(:, roi_idx) = squeeze(mean(mean(block, 1), 2));
end
end
