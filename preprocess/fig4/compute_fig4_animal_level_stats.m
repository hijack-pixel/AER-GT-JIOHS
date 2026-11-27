function stats = compute_fig4_animal_level_stats(metrics)
%COMPUTE_FIG4_ANIMAL_LEVEL_STATS Animal-level statistics for Fig. 4 metrics.
%
% With one 20-s segment per state each animal contributes exactly one
% observation per state, so the animal (not the window) is the independent
% statistical unit.
%
%   Omnibus : Friedman test per metric.
%   Pairwise: Wilcoxon signed-rank with Holm correction (family = 3 pairs).
%   Effect  : Kendall's W for each Friedman test.

state_labels = metrics.StateLabels;
num_mice = numel(metrics.MouseNames);
num_states = numel(state_labels);
pair_idx = [1, 2; 1, 3; 2, 3];
pair_names = {'baseline_vs_seizure', 'baseline_vs_recovery', 'seizure_vs_recovery'};

sig_vals = zeros(num_mice, num_states);
dev_vals = zeros(num_mice, num_states);
for mouse_idx = 1:num_mice
    for state_idx = 1:num_states
        sig_vals(mouse_idx, state_idx) = mean(metrics.MouseWindowMetric{mouse_idx, state_idx}, 'omitnan');
        dev_vals(mouse_idx, state_idx) = mean(metrics.MouseNetworkDeviation{mouse_idx, state_idx}, 'omitnan');
    end
end

% friedman() opens an ANOVA-table figure per call; close the ones we open.
figs_before = findall(0, 'Type', 'figure');
p_friedman_sig = friedman(sig_vals, 1);
p_friedman_dev = friedman(dev_vals, 1);
figs_opened = setdiff(findall(0, 'Type', 'figure'), figs_before);
if ~isempty(figs_opened)
    delete(figs_opened);
end

p_raw_sig = zeros(1, 3);
p_raw_dev = zeros(1, 3);
for pair = 1:3
    i = pair_idx(pair, 1);
    j = pair_idx(pair, 2);
    p_raw_sig(pair) = signrank(sig_vals(:, i), sig_vals(:, j));
    p_raw_dev(pair) = signrank(dev_vals(:, i), dev_vals(:, j));
end

stats = struct();
stats.MetricNames = {'window_mean_signal', 'network_deviation'};
stats.SignalPerMouse = sig_vals;
stats.DeviationPerMouse = dev_vals;
stats.FriedmanP = [p_friedman_sig, p_friedman_dev];
stats.KendallW = [local_kendall_w(sig_vals), local_kendall_w(dev_vals)];
stats.PairNames = pair_names;
stats.PRawSignal = p_raw_sig;
stats.PRawDeviation = p_raw_dev;
stats.PAdjSignal = local_holm(p_raw_sig);
stats.PAdjDeviation = local_holm(p_raw_dev);

fprintf('\nFig.4 animal-level statistics (n = %d mice):\n', num_mice);
fprintf('  Friedman omnibus: signal p = %.4f (W = %.2f), deviation p = %.4f (W = %.2f)\n', ...
    p_friedman_sig, stats.KendallW(1), p_friedman_dev, stats.KendallW(2));
for pair = 1:3
    fprintf('  %-22s signal p_raw = %.4f p_adj = %.4f | deviation p_raw = %.4f p_adj = %.4f\n', ...
        pair_names{pair}, p_raw_sig(pair), stats.PAdjSignal(pair), ...
        p_raw_dev(pair), stats.PAdjDeviation(pair));
end
end

function w = local_kendall_w(values)
num_blocks = size(values, 1);
num_treatments = size(values, 2);
rank_sums = zeros(1, num_treatments);
for block_idx = 1:num_blocks
    rank_sums = rank_sums + tiedrank(values(block_idx, :));
end
expected = num_blocks * (num_treatments + 1) / 2;
w = 12 * sum((rank_sums - expected).^2) / ...
    (num_blocks^2 * (num_treatments^3 - num_treatments));
end

function p_adj = local_holm(p_raw)
[p_sorted, order] = sort(p_raw(:));
family = numel(p_sorted);
p_adj = nan(family, 1);
running_max = 0;
for k = 1:family
    running_max = max(running_max, (family - k + 1) * p_sorted(k));
    p_adj(order(k)) = min(1, running_max);
end
p_adj = p_adj';
end
