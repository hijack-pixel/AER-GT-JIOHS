function roi_map = aergt_map_epilepsy_rois(mouse_points, fixed_points, roi_positions, image_size, box_half)
%AERGT_MAP_EPILEPSY_ROIS Project standard ROI coordinates into a mouse frame.

tform = cp2tform(mouse_points, fixed_points, 'nonreflective similarity');
Tinv = tform.tdata.Tinv;
roi_map = floor(roi_positions * Tinv(1:2, 1:2) + repmat(Tinv(3, 1:2), size(roi_positions, 1), 1));
roi_map = aergt_clip_roi_map(roi_map, image_size, box_half);
end
