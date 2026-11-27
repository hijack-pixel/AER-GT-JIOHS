function export_control_samples(cfg)
%EXPORT_CONTROL_SAMPLES Export unlabeled samples of the pharmacological-control cohort.
%
% The seizure-free pilocarpine-injected control recordings (Sec. 2.9) are
% processed with the pipeline of the labeled cohort and saved with Label = -1;
% they are never used for training.
%
% Example:
%   cfg = aergt_preprocess_config_template();
%   export_control_samples(cfg);

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

aergt_export_unlabeled_cohort(cfg, 'control', cfg.output.control_dir);
end
