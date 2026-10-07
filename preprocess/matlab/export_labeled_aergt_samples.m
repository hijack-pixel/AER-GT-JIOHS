function export_labeled_aergt_samples(cfg)
%EXPORT_LABELED_AERGT_SAMPLES Export labeled epilepsy graph samples.
%
% Output files contain one variable, AERT_Sample. Labels are:
%   0 = baseline, 1 = seizure, 2 = post-ictal recovery.
%
% Example:
%   cfg = aergt_preprocess_config_template();
%   cfg.raw.epilepsy_root = 'PATH_TO_PRIVATE_EPILEPSY_MOUSE_DATA';
%   export_labeled_aergt_samples(cfg);

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

output_dir = cfg.output.labeled_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fps = cfg.signal.fps;
win_frames = cfg.signal.win_sec * fps;
step_frames = cfg.signal.step_sec * fps;
box_half = cfg.signal.box_half;

Mice = aergt_build_paper_mouse_table(cfg);
Mice = aergt_apply_mouse_recording_spec(Mice);

fprintf('Exporting labeled AER-GT samples to %s\n', output_dir);

for mouse_idx = 1:numel(Mice)
    mouse = Mice(mouse_idx);
    if ~exist(mouse.Folder, 'dir')
        error('export_labeled_aergt_samples:MissingFolder', ...
            'Raw folder not found for %s: %s', mouse.Name, mouse.Folder);
    end

    base_path = fullfile(mouse.Folder, mouse.BaseFile);
    if ~exist(base_path, 'file')
        error('export_labeled_aergt_samples:MissingBaseFile', ...
            'Base file not found for %s: %s', mouse.Name, base_path);
    end

    base_info = imfinfo(base_path);
    image_size = [base_info(1).Height, base_info(1).Width];
    roi_map = aergt_map_epilepsy_rois( ...
        mouse.Coords, ...
        cfg.registration.fixed_points, ...
        cfg.registration.roi_positions, ...
        image_size, ...
        box_half ...
    );

    base_video = aergt_read_tiff_stack(base_path);
    base_roi = aergt_extract_roi_timeseries(base_video, roi_map, box_half);
    clear base_video;

    base_window = min(cfg.signal.baseline_frames, size(base_roi, 1));
    mouse_baseline_vector = mean(base_roi(1:base_window, :), 1);

    fprintf('\n[%s] %d paper recordings\n', mouse.Name, numel(mouse.Files));

    for file_idx = 1:numel(mouse.Files)
        current_file = mouse.Files{file_idx};
        full_path = fullfile(mouse.Folder, current_file);
        if ~exist(full_path, 'file')
            error('export_labeled_aergt_samples:MissingRecording', ...
                'Missing paper recording for %s: %s', mouse.Name, full_path);
        end

        label_id = aergt_label_from_filename(current_file);
        if label_id == -1
            error('export_labeled_aergt_samples:UnlabeledFile', ...
                'Unable to assign label for %s (%s).', mouse.Name, current_file);
        end

        video = aergt_read_tiff_stack(full_path);
        roi_traces = aergt_extract_roi_timeseries(video, roi_map, box_half);
        clear video;

        normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);
        num_frames = size(normalized_traces, 1);
        num_windows = floor((num_frames - win_frames) / step_frames) + 1;
        if num_windows < 1
            warning('export_labeled_aergt_samples:ShortRecording', ...
                'Skipping %s (%s): only %d frames.', mouse.Name, current_file, num_frames);
            continue;
        end

        fprintf('  %s | label=%d | frames=%d | windows=%d\n', ...
            current_file, label_id, num_frames, num_windows);

        for window_idx = 1:num_windows
            idx_start = (window_idx - 1) * step_frames + 1;
            idx_end = idx_start + win_frames - 1;
            clip = normalized_traces(idx_start:idx_end, :);

            AERT_Sample = struct();
            AERT_Sample.MouseID = mouse.Name;
            AERT_Sample.OriginalFile = current_file;
            AERT_Sample.Label = label_id;
            AERT_Sample.Adjacency = aergt_compute_adjacency(clip);
            AERT_Sample.NodeFeats = aergt_compute_6d_features( ...
                clip, fps, cfg.signal.low_band_hz, cfg.signal.high_band_hz ...
            );
            AERT_Sample.WindowIndex = window_idx;
            AERT_Sample.WindowStartFrame = idx_start;
            AERT_Sample.WindowEndFrame = idx_end;
            AERT_Sample.FPS = fps;
            AERT_Sample.WinSec = cfg.signal.win_sec;
            AERT_Sample.StepSec = cfg.signal.step_sec;
            AERT_Sample.ROIMap = roi_map;
            AERT_Sample.ROI_Names = cfg.registration.roi_labels;
            AERT_Sample.MouseBaseline = mouse_baseline_vector;

            clean_mouse = aergt_safe_filename(mouse.Name);
            clean_file = aergt_safe_filename(current_file);
            save_name = fullfile(output_dir, sprintf('%s_%s_Win%d.mat', clean_mouse, clean_file, window_idx));

            if cfg.output.overwrite || ~exist(save_name, 'file')
                save(save_name, 'AERT_Sample');
            end
        end
    end
end

fprintf('\nLabeled AER-GT sample export complete.\n');
end
