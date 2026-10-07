function metrics = compute_fig4_ios_state_metrics(cfg)
%COMPUTE_FIG4_IOS_STATE_METRICS Compute IOS state traces and FC deviation.
%
% The analysis follows the manuscript Fig.4 preprocessing path:
%   - five epilepsy mice
%   - one representative baseline, seizure, and recovery recording per mouse
%   - first 300 frames per recording
%   - dI/I normalization by each file's first 100 frames
%   - frames 101:300 with a mild median filter
%
% Corrections applied in response to reviewer comments:
%   1. Window length is 200 frames = 20 s at 10 Hz, matching the manuscript
%      (the initial release used 40 frames = 4 s).
%   2. Functional connectivity is computed per animal and then averaged
%      across animals (the initial release correlated the cross-animal
%      averaged time series).
%   3. Network deviation is evaluated at the animal level: with 20-s windows
%      each mouse contributes exactly one observation per state.

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

state_labels = {'Baseline', 'Seizure', 'Recovery'};
state_files = local_fig4_state_files();

fps = cfg.signal.fps;
box_half = cfg.signal.box_half;
fig4_win_len = 200;  % 20 s at 10 Hz (manuscript description)
fig4_win_step = 40;  % 4 s stride; 200-frame clips give one window per state
main_roi_idx = 15;
deviation_metric = 'corrdist';

Mice = aergt_build_paper_mouse_table(cfg);
if numel(Mice) ~= size(state_files, 1)
    error('compute_fig4_ios_state_metrics:MouseCountMismatch', ...
        'Expected %d mice, got %d.', size(state_files, 1), numel(Mice));
end

num_mice = numel(Mice);
num_states = numel(state_labels);
mouse_clips = cell(num_mice, num_states);
mouse_state_trace = cell(num_mice, num_states);
mouse_display_trace = cell(num_mice, num_states);
mouse_window_metric = cell(num_mice, num_states);
mouse_adjacency = cell(num_mice, num_states);

for mouse_idx = 1:num_mice
    mouse = Mice(mouse_idx);
    baseline_path = fullfile(mouse.Folder, state_files{mouse_idx, 1});
    if ~exist(baseline_path, 'file')
        error('compute_fig4_ios_state_metrics:MissingBaseline', ...
            'Missing baseline file for %s: %s', mouse.Name, baseline_path);
    end

    base_info = imfinfo(baseline_path);
    image_size = [base_info(1).Height, base_info(1).Width];
    roi_map = aergt_map_epilepsy_rois( ...
        mouse.Coords, ...
        cfg.registration.fixed_points, ...
        cfg.registration.roi_positions, ...
        image_size, ...
        box_half ...
    );

    for state_idx = 1:num_states
        file_path = fullfile(mouse.Folder, state_files{mouse_idx, state_idx});
        if ~exist(file_path, 'file')
            error('compute_fig4_ios_state_metrics:MissingStateFile', ...
                'Missing state file for %s: %s', mouse.Name, file_path);
        end

        video = aergt_read_tiff_stack(file_path, 300);
        if size(video, 3) < 300
            error('compute_fig4_ios_state_metrics:ShortRecording', ...
                'Fig.4 requires at least 300 frames: %s.', file_path);
        end

        roi_traces = aergt_extract_roi_timeseries(video, roi_map, box_half);
        clear video;

        normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);
        clip = medfilt1(normalized_traces(101:300, :), 3, [], 1);

        mouse_clips{mouse_idx, state_idx} = clip;
        mouse_state_trace{mouse_idx, state_idx} = mean(clip, 2, 'omitnan');
        mouse_display_trace{mouse_idx, state_idx} = mean(clip(:, main_roi_idx), 2, 'omitnan');

        % Per-animal connectivity over the full (single 20-s) window
        mouse_adjacency{mouse_idx, state_idx} = aergt_compute_adjacency(clip);

        starts = 1:fig4_win_step:(size(clip, 1) - fig4_win_len + 1);
        metric_vals = zeros(numel(starts), 1);
        for window_idx = 1:numel(starts)
            idx = starts(window_idx):(starts(window_idx) + fig4_win_len - 1);
            metric_vals(window_idx) = mean(clip(idx, :), 'all', 'omitnan');
        end
        mouse_window_metric{mouse_idx, state_idx} = metric_vals;
    end
end

% Cross-animal averaged clips are used for DISPLAY TRACES only.
% Functional connectivity is computed per animal and then averaged
% (corrected order: the initial release correlated the averaged series).
state_average_clips = cell(1, num_states);
state_trace = cell(1, num_states);
state_display_trace = cell(1, num_states);
adjacency_matrices = cell(1, num_states);
window_metric = cell(1, num_states);

for state_idx = 1:num_states
    state_average_clips{state_idx} = mean(cat(3, mouse_clips{:, state_idx}), 3, 'omitnan');
    clip = state_average_clips{state_idx};
    state_trace{state_idx} = mean(clip, 2, 'omitnan');
    state_display_trace{state_idx} = mean(clip(:, main_roi_idx), 2, 'omitnan');
    window_metric{state_idx} = vertcat(mouse_window_metric{:, state_idx});

    % Connectivity: average of per-animal estimates (corrected)
    adjacency = mean(cat(3, mouse_adjacency{:, state_idx}), 3, 'omitnan');
    adjacency(1:size(adjacency, 1)+1:end) = 1;
    adjacency_matrices{state_idx} = adjacency;
end

% Network deviation at the animal level: with 20-s windows each mouse
% contributes exactly one deviation value per state.
mouse_network_deviation = local_mouse_network_deviation(mouse_clips, fig4_win_len, fig4_win_step, deviation_metric);
network_deviation = cell(1, num_states);
for state_idx = 1:num_states
    network_deviation{state_idx} = vertcat(mouse_network_deviation{:, state_idx});
end

metrics = struct();
metrics.StateLabels = state_labels;
metrics.MouseNames = {Mice.Name};
metrics.StateFiles = state_files;
metrics.ROI_Names = cfg.registration.roi_labels;
metrics.MouseClips = mouse_clips;
metrics.MouseStateTrace = mouse_state_trace;
metrics.MouseDisplayTrace = mouse_display_trace;
metrics.StateAverageClips = state_average_clips;
metrics.StateTrace = state_trace;
metrics.StateDisplayTrace = state_display_trace;
metrics.MouseAdjacency = mouse_adjacency;
metrics.AdjacencyMatrices = adjacency_matrices;
metrics.WindowMetric = window_metric;
metrics.MouseWindowMetric = mouse_window_metric;
metrics.NetworkDeviation = network_deviation;
metrics.MouseNetworkDeviation = mouse_network_deviation;
metrics.Parameters = struct( ...
    'fps', fps, ...
    'frames_per_state_file', 300, ...
    'analysis_frames', [101, 300], ...
    'window_length_frames', fig4_win_len, ...
    'window_step_frames', fig4_win_step, ...
    'display_roi_index', main_roi_idx, ...
    'deviation_metric', deviation_metric, ...
    'connectivity_order', 'per_animal_then_average', ...
    'deviation_unit', 'per_mouse' ...
);
end

function state_files = local_fig4_state_files()
state_files = {
    '10min.tif',           '1.5h.tif',          '4h.tif';
    '10min_transfer.tif',  '1h.tif',            '2.5h (6).tif';
    '10min.tif',           '40min.tif',         '2h.tif';
    '10min.tif',           '1h.tif',            '2h.tif';
    '10min.tif',           '20min_seizure.tif', '180.tif'
};
end

function mouse_deviation = local_mouse_network_deviation(mouse_clips, win_len, win_step, deviation_metric)
num_mice = size(mouse_clips, 1);
num_states = size(mouse_clips, 2);
mask = triu(true(22), 1);
mouse_deviation = cell(num_mice, num_states);

for mouse_idx = 1:num_mice
    baseline_clip = mouse_clips{mouse_idx, 1};
    baseline_adjacency = aergt_compute_adjacency(baseline_clip);
    baseline_vector = baseline_adjacency(mask);

    for state_idx = 1:num_states
        clip = mouse_clips{mouse_idx, state_idx};
        starts = 1:win_step:(size(clip, 1) - win_len + 1);
        values = zeros(numel(starts), 1);
        for window_idx = 1:numel(starts)
            idx = starts(window_idx):(starts(window_idx) + win_len - 1);
            current_adjacency = aergt_compute_adjacency(clip(idx, :));
            switch lower(deviation_metric)
                case 'corrdist'
                    current_vector = current_adjacency(mask);
                    values(window_idx) = 1 - corr(current_vector, baseline_vector, 'Rows', 'complete');
                case 'fro'
                    values(window_idx) = norm(current_adjacency - baseline_adjacency, 'fro');
                otherwise
                    error('Unknown deviation metric: %s.', deviation_metric);
            end
        end
        mouse_deviation{mouse_idx, state_idx} = values;
    end
end
end
