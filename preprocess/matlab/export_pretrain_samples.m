function export_pretrain_samples(cfg)
%EXPORT_PRETRAIN_SAMPLES Export unlabeled samples of the normative-baseline cohort.
%
% The NB1 and NB2 saline-control recordings are used only for self-supervised
% pretraining (and for fitting the feature standardization) and are saved with
% Label = -1.
%
% Example:
%   cfg = aergt_preprocess_config_template();
%   export_pretrain_samples(cfg);

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

aergt_export_unlabeled_cohort(cfg, 'pretrain', cfg.output.pretrain_dir);
end
