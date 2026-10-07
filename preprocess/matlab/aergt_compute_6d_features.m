function node_features = aergt_compute_6d_features(clip, fps, low_band_hz, high_band_hz)
%AERGT_COMPUTE_6D_FEATURES Compute the paper's six ROI features.

num_rois = size(clip, 2);

feat_mean = mean(clip, 1);
feat_std = std(clip, 0, 1);
feat_skew = skewness(clip, 0, 1);
feat_kurt = kurtosis(clip, 0, 1);

feat_low_freq = zeros(1, num_rois);
feat_high_freq = zeros(1, num_rois);

for roi_idx = 1:num_rois
    signal = clip(:, roi_idx);
    feat_low_freq(roi_idx) = bandpower(signal, fps, low_band_hz);
    feat_high_freq(roi_idx) = bandpower(signal, fps, high_band_hz);
end

node_features = [
    feat_mean', ...
    feat_std', ...
    feat_skew', ...
    feat_kurt', ...
    feat_low_freq', ...
    feat_high_freq'
];
end
