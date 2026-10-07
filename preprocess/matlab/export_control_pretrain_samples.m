function export_control_pretrain_samples(cfg)
%EXPORT_CONTROL_PRETRAIN_SAMPLES Export unlabeled CON1+CON2 pretraining samples.
%
% CON1 and CON2 are saline-control batches. They are used only for
% self-supervised pretraining and are saved with Label = -1.
%
% Example:
%   cfg = aergt_preprocess_config_template();
%   cfg.raw.control_con1_root = 'PATH_TO_PRIVATE_CON1';
%   cfg.raw.control_con2_root = 'PATH_TO_PRIVATE_CON2';
%   export_control_pretrain_samples(cfg);

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

output_dir = cfg.output.pretrain_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fps = cfg.signal.fps;
win_frames = cfg.signal.win_sec * fps;
step_frames = cfg.signal.step_sec * fps;
box_half = cfg.signal.box_half;

specs = aergt_control_specs();

fprintf('Exporting CON1+CON2 control pretraining samples to %s\n', output_dir);

for spec_idx = 1:numel(specs)
    spec = specs(spec_idx);
    if strcmpi(spec.Name, 'CON1')
        raw_folder = cfg.raw.control_con1_root;
    else
        raw_folder = cfg.raw.control_con2_root;
    end

    if ~exist(raw_folder, 'dir')
        error('export_control_pretrain_samples:MissingFolder', ...
            '%s folder not found: %s', spec.Name, raw_folder);
    end

    fprintf('\n[%s] %d saline-control mice\n', spec.Name, spec.NumMice);

    for mouse_idx = 1:spec.NumMice
        mouse_id = sprintf(spec.MouseIdPattern, mouse_idx);
        file_names = aergt_find_control_files(raw_folder, spec.Name, mouse_idx);
        if isempty(file_names)
            warning('export_control_pretrain_samples:MissingMouseFiles', ...
                'No TIFF chunks found for %s.', mouse_id);
            continue;
        end

        fprintf('  %s | files=%d\n', mouse_id, numel(file_names));

        video = aergt_read_tiff_sequence(raw_folder, file_names);
        image_size = [size(video, 1), size(video, 2)];
        mouse_points = [
            spec.MousePoints(mouse_idx, 1), spec.MousePoints(mouse_idx, 2);
            spec.MousePoints(mouse_idx, 3), spec.MousePoints(mouse_idx, 4)
        ];

        roi_map = aergt_map_control_rois( ...
            mouse_points, ...
            cfg.registration.fixed_points, ...
            cfg.registration.roi_positions, ...
            image_size, ...
            box_half ...
        );

        roi_traces = aergt_extract_roi_timeseries(video, roi_map, box_half);
        clear video;

        normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);

        if cfg.output.write_quality_reports
            quality_dir = fullfile(cfg.output.quality_dir, spec.Name);
            aergt_write_control_quality_report( ...
                quality_dir, mouse_id, roi_traces, normalized_traces, fps, cfg.registration.roi_labels ...
            );
        end

        total_frames = size(normalized_traces, 1);
        num_windows = floor((total_frames - win_frames) / step_frames) + 1;
        if num_windows < 1
            warning('export_control_pretrain_samples:ShortRecording', ...
                'Skipping %s: only %d frames.', mouse_id, total_frames);
            continue;
        end

        for window_idx = 1:num_windows
            idx_start = (window_idx - 1) * step_frames + 1;
            idx_end = idx_start + win_frames - 1;
            clip = normalized_traces(idx_start:idx_end, :);

            AERT_Sample = struct();
            AERT_Sample.MouseID = mouse_id;
            AERT_Sample.ControlSet = spec.Name;
            AERT_Sample.OriginalFile = strjoin(file_names, ' | ');
            AERT_Sample.Label = -1;
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

            save_name = fullfile(output_dir, sprintf('%s_Win%04d.mat', aergt_safe_filename(mouse_id), window_idx));
            if cfg.output.overwrite || ~exist(save_name, 'file')
                save(save_name, 'AERT_Sample');
            end
        end

        fprintf('    windows=%d\n', num_windows);
    end
end

fprintf('\nCON1+CON2 control pretraining sample export complete.\n');
end
