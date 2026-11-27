function M = aergt_transform_matrix(tform)
%AERGT_TRANSFORM_MATRIX Forward matrix of a 2-D geometric transform.
%
% Returns the 3 x 3 matrix M in the row-vector convention, [x y 1] * M = [u v 1],
% for the transform objects returned by fitgeotrans (affine2d and
% projective2d store it in T; affinetform2d stores the transposed matrix in A).

if isa(tform, 'affinetform2d')
    M = tform.A.';
else
    M = tform.T;
end
end
