function normalized_traces = aergt_normalize_roi_traces(roi_traces, baseline_frames)
%AERGT_NORMALIZE_ROI_TRACES Compute percent intensity change per ROI.

num_frames = size(roi_traces, 1);
baseline_window = min(baseline_frames, num_frames);
baseline = mean(roi_traces(1:baseline_window, :), 1);
baseline(baseline == 0) = eps;
normalized_traces = 100 * (roi_traces - repmat(baseline, num_frames, 1)) ./ repmat(baseline, num_frames, 1);
end
