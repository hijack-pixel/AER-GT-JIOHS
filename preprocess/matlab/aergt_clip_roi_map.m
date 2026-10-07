function roi_map = aergt_clip_roi_map(roi_map, image_size, box_half)
%AERGT_CLIP_ROI_MAP Keep ROI boxes inside image boundaries.

height = image_size(1);
width = image_size(2);

roi_map(:, 1) = max(1 + box_half, min(roi_map(:, 1), width - box_half));
roi_map(:, 2) = max(1 + box_half, min(roi_map(:, 2), height - box_half));
end
