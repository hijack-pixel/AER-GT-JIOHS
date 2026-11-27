function export_labeled_aergt_samples(cfg)
%EXPORT_LABELED_AERGT_SAMPLES Export labeled graph samples of the ictal-transition cohort.
%
% Every recording listed for the 'labeled' cohort in cfg.spec.recordings is
% processed with the pipeline of Sec. 2.2: the 22 ROIs are placed with the
% animal's registration landmarks, the ROI time series are normalized against
% the first cfg.signal.baseline_frames frames, and windows of
% cfg.signal.win_sec seconds are taken with a stride of cfg.signal.step_sec
% seconds (a recording of N frames yields floor((N - win_frames) / step_frames)
% + 1 windows). Each output file contains one variable, AERT_Sample; the
% reference label of the recording is assigned at the session level:
%   0 = baseline, 1 = seizure, 2 = post-ictal recovery.
%
% Example:
%   cfg = aergt_preprocess_config_template();
%   export_labeled_aergt_samples(cfg);

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

output_dir = cfg.output.labeled_dir;
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

Animals = aergt_read_animal_table(cfg, 'labeled');
Recordings = aergt_read_recording_table(cfg, 'labeled', Animals);

fprintf('Exporting labeled samples to %s\n', output_dir);

for r = 1:numel(Recordings)
    rec = Recordings(r);
    mouse = Animals(strcmpi({Animals.Name}, rec.Animal));
    if ~exist(mouse.Folder, 'dir')
        error('export_labeled_aergt_samples:MissingFolder', ...
            'Raw folder not found for %s: %s', mouse.Name, mouse.Folder);
    end
    full_path = fullfile(mouse.Folder, rec.Files{1});
    if ~exist(full_path, 'file')
        error('export_labeled_aergt_samples:MissingRecording', ...
            'Missing recording for %s: %s', mouse.Name, full_path);
    end

    info = imfinfo(full_path);
    image_size = [info(1).Height, info(1).Width];
    roi_map = aergt_map_rois_similarity( ...
        mouse.Coords, cfg.registration.fixed_points, cfg.registration.roi_positions, ...
        image_size, cfg.signal.box_half);

    video = aergt_read_tiff_stack(full_path);
    roi_traces = aergt_extract_roi_timeseries(video, roi_map, cfg.signal.box_half);
    clear video;

    normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);
    file_stem = sprintf('%s_%s', aergt_safe_filename(mouse.Name), aergt_safe_filename(rec.Files{1}));
    n_windows = aergt_write_window_samples(cfg, normalized_traces, roi_map, ...
        mouse.Name, rec.Files{1}, rec.Label, output_dir, file_stem);

    if n_windows < 1
        warning('export_labeled_aergt_samples:ShortRecording', ...
            'Skipping %s (%s): only %d frames.', mouse.Name, rec.Files{1}, size(normalized_traces, 1));
        continue;
    end
    fprintf('  %s | %s | label=%d | frames=%d | windows=%d\n', ...
        mouse.Name, rec.Files{1}, rec.Label, size(normalized_traces, 1), n_windows);
end

fprintf('Labeled sample export complete.\n');
end
