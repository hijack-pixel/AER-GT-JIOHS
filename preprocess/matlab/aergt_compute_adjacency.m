function adjacency = aergt_compute_adjacency(clip)
%AERGT_COMPUTE_ADJACENCY Compute ROI functional connectivity for a window.

adjacency = corr(clip);
adjacency(isnan(adjacency)) = 0;
adjacency(1:size(adjacency, 1)+1:end) = 1;
end
