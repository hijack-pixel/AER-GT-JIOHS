function cfg = aergt_preprocess_config_template()
%AERGT_PREPROCESS_CONFIG_TEMPLATE Return default preprocessing configuration.
%
% Edit only the raw-data paths below for a private full-data run. The default
% output paths are under data/private/ and are excluded by .gitignore.

this_file = mfilename('fullpath');
matlab_dir = fileparts(this_file);
preprocess_dir = fileparts(matlab_dir);
repo_root = fileparts(preprocess_dir);

cfg = struct();

cfg.raw = struct();
cfg.raw.epilepsy_root = fullfile(repo_root, 'data', 'private', 'raw_epilepsy');
cfg.raw.control_con1_root = fullfile(repo_root, 'data', 'private', 'raw_control', 'CON1');
cfg.raw.control_con2_root = fullfile(repo_root, 'data', 'private', 'raw_control', 'CON2');

cfg.output = struct();
cfg.output.labeled_dir = fullfile(repo_root, 'data', 'private', 'labeled_epilepsy');
cfg.output.pretrain_dir = fullfile(repo_root, 'data', 'private', 'pretrain_con1_con2');
cfg.output.quality_dir = fullfile(repo_root, 'outputs', 'preprocess_quality');
cfg.output.overwrite = true;
cfg.output.write_quality_reports = false;

cfg.signal = struct();
cfg.signal.fps = 10;
cfg.signal.win_sec = 20;
cfg.signal.step_sec = 4;
cfg.signal.baseline_frames = 100;
cfg.signal.box_half = 5;
cfg.signal.low_band_hz = [0.01, 0.1];
cfg.signal.high_band_hz = [0.1, 0.5];

cfg.registration = struct();
cfg.registration.fixed_points = [143, 132; 143, 240];
cfg.registration.roi_positions = aergt_default_roi_positions();
cfg.registration.roi_labels = aergt_default_roi_labels();
end
