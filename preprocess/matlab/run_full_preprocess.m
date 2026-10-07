function run_full_preprocess(cfg)
%RUN_FULL_PREPROCESS Run the private-data preprocessing workflow.
%
% This function exports:
%   1. CON1+CON2 saline-control samples for self-supervised pretraining.
%   2. Labeled epilepsy samples for baseline/seizure/recovery training.

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

export_control_pretrain_samples(cfg);
validate_aert_sample_folder(cfg.output.pretrain_dir, false);

export_labeled_aergt_samples(cfg);
validate_aert_sample_folder(cfg.output.labeled_dir, true);

fprintf('\nFull AER-GT preprocessing completed.\n');
fprintf('Pretraining samples: %s\n', cfg.output.pretrain_dir);
fprintf('Labeled samples:     %s\n', cfg.output.labeled_dir);
end
