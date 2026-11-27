function roi_map = aergt_map_rois_similarity(mouse_points, fixed_points, roi_positions, image_size, box_half)
%AERGT_MAP_ROIS_SIMILARITY Project the template ROI positions into a recording.
%
% A non-reflective similarity transform maps the animal's registration landmarks
% (mouse_points) onto the two landmarks of the common cortical template
% (fixed_points); the template ROI positions are mapped with its inverse into the
% recording and kept inside the image.

tform = fitgeotrans(mouse_points, fixed_points, 'nonreflectivesimilarity');
inverse = inv(aergt_transform_matrix(tform));
roi_map = floor(roi_positions * inverse(1:2, 1:2) + repmat(inverse(3, 1:2), size(roi_positions, 1), 1));
roi_map = aergt_clip_roi_map(roi_map, image_size, box_half);
end
