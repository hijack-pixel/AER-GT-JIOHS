function n_windows = aergt_write_window_samples(cfg, normalized_traces, roi_map, mouse_id, original_file, label_id, output_dir, file_stem)
%AERGT_WRITE_WINDOW_SAMPLES Cut a recording into windows and save one AERT_Sample per window.
%
%   n_windows = aergt_write_window_samples(cfg, normalized_traces, roi_map, ...
%       mouse_id, original_file, label_id, output_dir, file_stem)
%
% Windows of cfg.signal.win_sec seconds are taken with a stride of
% cfg.signal.step_sec seconds, so that a recording of N frames yields
% floor((N - win_frames) / step_frames) + 1 windows. Each window is saved as
% <file_stem>_Win<k>.mat with a single variable AERT_Sample:
%   MouseID, OriginalFile, Label, Adjacency (22 x 22), NodeFeats (22 x 6),
%   WindowIndex, WindowStartFrame, WindowEndFrame, FPS, WinSec, StepSec,
%   ROIMap, ROI_Names.

fps = cfg.signal.fps;
win_frames = round(cfg.signal.win_sec * fps);
step_frames = round(cfg.signal.step_sec * fps);

num_frames = size(normalized_traces, 1);
n_windows = floor((num_frames - win_frames) / step_frames) + 1;

for window_idx = 1:max(n_windows, 0)
    idx_start = (window_idx - 1) * step_frames + 1;
    idx_end = idx_start + win_frames - 1;
    clip = normalized_traces(idx_start:idx_end, :);

    AERT_Sample = struct();
    AERT_Sample.MouseID = mouse_id;
    AERT_Sample.OriginalFile = original_file;
    AERT_Sample.Label = label_id;
    AERT_Sample.Adjacency = aergt_compute_adjacency(clip);
    AERT_Sample.NodeFeats = aergt_compute_6d_features( ...
        clip, fps, cfg.signal.low_band_hz, cfg.signal.high_band_hz);
    AERT_Sample.WindowIndex = window_idx;
    AERT_Sample.WindowStartFrame = idx_start;
    AERT_Sample.WindowEndFrame = idx_end;
    AERT_Sample.FPS = fps;
    AERT_Sample.WinSec = cfg.signal.win_sec;
    AERT_Sample.StepSec = cfg.signal.step_sec;
    AERT_Sample.ROIMap = roi_map;
    AERT_Sample.ROI_Names = cfg.registration.roi_labels;

    save_name = fullfile(output_dir, sprintf('%s_Win%d.mat', file_stem, window_idx));
    if cfg.output.overwrite || ~exist(save_name, 'file')
        save(save_name, 'AERT_Sample');
    end
end
end
