function run_fig4_ios_state_analysis(cfg)
%RUN_FIG4_IOS_STATE_ANALYSIS Compute Fig.4 IOS state-analysis metrics.
%
% This entry writes numerical results only. Generated figures are excluded
% from the repository and should be recreated from the saved metrics.

fig4_dir = fileparts(mfilename('fullpath'));
preprocess_dir = fileparts(fig4_dir);
addpath(fullfile(preprocess_dir, 'matlab'));

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

metrics = compute_fig4_ios_state_metrics(cfg);
stats = compute_fig4_animal_level_stats(metrics);

repo_root = fileparts(preprocess_dir);
output_dir = fullfile(repo_root, 'outputs', 'fig4_ios_state_analysis');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

save(fullfile(output_dir, 'fig4_ios_state_metrics.mat'), 'metrics');
save(fullfile(output_dir, 'fig4_animal_level_stats.mat'), 'stats');
fprintf('Saved Fig.4 IOS state-analysis metrics to %s\n', output_dir);
fprintf('Saved Fig.4 animal-level statistics (Friedman/Wilcoxon-Holm) to %s\n', output_dir);
end
