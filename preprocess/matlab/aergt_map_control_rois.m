function roi_map = aergt_map_control_rois(mouse_points, fixed_points, roi_positions, image_size, box_half)
%AERGT_MAP_CONTROL_ROIS Project standard ROI coordinates into a control frame.

std_dist_y = norm(fixed_points(1, :) - fixed_points(2, :));
mouse_dist_y = norm(mouse_points(1, :) - mouse_points(2, :));

scale_y = mouse_dist_y / std_dist_y;
scale_x = scale_y;

vec_std = fixed_points(2, :) - fixed_points(1, :);
vec_std_perp = [vec_std(2), -vec_std(1)];
fixed_point_3 = fixed_points(1, :) + vec_std_perp;

vec_mouse = mouse_points(2, :) - mouse_points(1, :);
vec_mouse_perp = [vec_mouse(2), -vec_mouse(1)];
mouse_point_3 = mouse_points(1, :) + vec_mouse_perp * (scale_x / scale_y);

tform = cp2tform([fixed_points; fixed_point_3], [mouse_points; mouse_point_3], 'affine');
TinM = tform.tdata.T;

roi_map = floor(roi_positions * TinM(1:2, 1:2) + repmat(TinM(3, 1:2), size(roi_positions, 1), 1));
roi_map = aergt_clip_roi_map(roi_map, image_size, box_half);
end
