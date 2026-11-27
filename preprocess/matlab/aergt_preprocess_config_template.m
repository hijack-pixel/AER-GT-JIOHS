function cfg = aergt_preprocess_config_template()
%AERGT_PREPROCESS_CONFIG_TEMPLATE Return default preprocessing configuration.
%
% For a private-data run, set the three raw-data roots and point cfg.spec.* at
% copies of the specification tables in preprocess/spec/ that list your animals
% (folders and registration landmarks) and recordings (files, chunk order, and
% reference labels). The default output paths are under data/private/ and are
% excluded by .gitignore.

this_file = mfilename('fullpath');
matlab_dir = fileparts(this_file);
preprocess_dir = fileparts(matlab_dir);
repo_root = fileparts(preprocess_dir);

cfg = struct();

cfg.raw = struct();
cfg.raw.epilepsy_root = fullfile(repo_root, 'data', 'private', 'raw_epilepsy');
cfg.raw.pretrain_root = fullfile(repo_root, 'data', 'private', 'raw_pretrain');
cfg.raw.control_root = fullfile(repo_root, 'data', 'private', 'raw_control');

cfg.spec = struct();
cfg.spec.animals = fullfile(preprocess_dir, 'spec', 'animals.csv');
cfg.spec.recordings = fullfile(preprocess_dir, 'spec', 'recordings.csv');
cfg.spec.fig4_state_recordings = fullfile(preprocess_dir, 'spec', 'fig4_state_recordings.csv');

cfg.output = struct();
cfg.output.labeled_dir = fullfile(repo_root, 'data', 'private', 'labeled_epilepsy');
cfg.output.pretrain_dir = fullfile(repo_root, 'data', 'private', 'pretrain_normative_baseline');
cfg.output.control_dir = fullfile(repo_root, 'data', 'private', 'control_pharmacological');
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

cfg.fig4 = struct();
cfg.fig4.display_roi_index = 15;
end
