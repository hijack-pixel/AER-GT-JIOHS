function run_full_preprocess(cfg)
%RUN_FULL_PREPROCESS Run the private-data preprocessing workflow.
%
% This function exports:
%   1. unlabeled samples of the normative-baseline (NB1 and NB2 saline) cohort,
%      used for self-supervised pretraining;
%   2. unlabeled samples of the pharmacological-control cohort;
%   3. labeled samples of the ictal-transition cohort.
%
% Animals, recordings, and labels are read from the specification tables named
% in cfg.spec (see aergt_preprocess_config_template).

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

export_pretrain_samples(cfg);
validate_aert_sample_folder(cfg.output.pretrain_dir, false);

export_control_samples(cfg);
validate_aert_sample_folder(cfg.output.control_dir, false);

export_labeled_aergt_samples(cfg);
validate_aert_sample_folder(cfg.output.labeled_dir, true);

fprintf('\nPreprocessing completed.\n');
fprintf('Pretraining samples:    %s\n', cfg.output.pretrain_dir);
fprintf('Control samples:        %s\n', cfg.output.control_dir);
fprintf('Labeled samples:        %s\n', cfg.output.labeled_dir);
end
