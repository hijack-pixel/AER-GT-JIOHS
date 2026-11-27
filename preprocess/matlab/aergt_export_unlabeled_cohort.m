function aergt_export_unlabeled_cohort(cfg, cohort, output_dir)
%AERGT_EXPORT_UNLABELED_COHORT Export unlabeled graph samples of one cohort.
%
%   aergt_export_unlabeled_cohort(cfg, 'pretrain', cfg.output.pretrain_dir)
%   aergt_export_unlabeled_cohort(cfg, 'control',  cfg.output.control_dir)
%
% 'pretrain' is the normative-baseline cohort (the NB1 and NB2 saline
% recordings used for self-supervised pretraining); 'control' is the
% pharmacological-control cohort. Both are processed with the pipeline of the
% labeled cohort (registration landmarks of each animal, ROI time series,
% normalization against the first frames, windows of cfg.signal.win_sec seconds
% with a stride of cfg.signal.step_sec seconds, six ROI features, correlation
% adjacency) and saved with Label = -1. A recording that consists of several
% TIFF chunks is read as one sequence (see aergt_read_recording_table).

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

Animals = aergt_read_animal_table(cfg, cohort);
Recordings = aergt_read_recording_table(cfg, cohort, Animals);
recordings_per_animal = zeros(1, numel(Animals));
for r = 1:numel(Recordings)
    recordings_per_animal(strcmpi({Animals.Name}, Recordings(r).Animal)) = ...
        recordings_per_animal(strcmpi({Animals.Name}, Recordings(r).Animal)) + 1;
end

fprintf('Exporting %s samples to %s\n', cohort, output_dir);

for r = 1:numel(Recordings)
    rec = Recordings(r);
    a = find(strcmpi({Animals.Name}, rec.Animal), 1);
    mouse = Animals(a);
    if ~exist(mouse.Folder, 'dir')
        error('aergt_export_unlabeled_cohort:MissingFolder', ...
            'Raw folder not found for %s: %s', mouse.Name, mouse.Folder);
    end
    for f = 1:numel(rec.Files)
        if ~exist(fullfile(mouse.Folder, rec.Files{f}), 'file')
            error('aergt_export_unlabeled_cohort:MissingFile', ...
                'Missing recording file for %s: %s', mouse.Name, rec.Files{f});
        end
    end

    if recordings_per_animal(a) > 1
        file_stem = sprintf('%s_%s', aergt_safe_filename(mouse.Name), aergt_safe_filename(rec.Recording));
    else
        file_stem = aergt_safe_filename(mouse.Name);
    end
    if numel(rec.Files) == 1
        original_file = rec.Files{1};
    else
        original_file = strjoin(rec.Files, ' | ');
    end

    video = aergt_read_tiff_sequence(mouse.Folder, rec.Files);
    image_size = [size(video, 1), size(video, 2)];
    roi_map = aergt_map_rois_affine( ...
        mouse.Coords, cfg.registration.fixed_points, cfg.registration.roi_positions, ...
        image_size, cfg.signal.box_half);
    roi_traces = aergt_extract_roi_timeseries(video, roi_map, cfg.signal.box_half);
    clear video;

    normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);

    if cfg.output.write_quality_reports
        aergt_write_quality_report(fullfile(cfg.output.quality_dir, cohort), file_stem, ...
            roi_traces, normalized_traces, cfg.signal.fps, cfg.registration.roi_labels);
    end

    n_windows = aergt_write_window_samples(cfg, normalized_traces, roi_map, ...
        mouse.Name, original_file, rec.Label, output_dir, file_stem);
    if n_windows < 1
        warning('aergt_export_unlabeled_cohort:ShortRecording', ...
            'Skipping %s (%s): only %d frames.', mouse.Name, original_file, size(normalized_traces, 1));
        continue;
    end
    fprintf('  %s | %s | frames=%d | windows=%d\n', ...
        mouse.Name, original_file, size(normalized_traces, 1), n_windows);
end

fprintf('%s sample export complete.\n', cohort);
end
