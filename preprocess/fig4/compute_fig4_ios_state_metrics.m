function metrics = compute_fig4_ios_state_metrics(cfg)
%COMPUTE_FIG4_IOS_STATE_METRICS Compute IOS state traces and connectivity deviation.
%
% State-level characterization of Fig. 4 (Sec. 2.3). For every animal one
% baseline-, one seizure-, and one recovery-labelled recording is read (the
% table cfg.spec.fig4_state_recordings) and processed as the labeled pipeline
% does: the ROIs are placed with the animal's registration landmarks, the ROI
% time series are normalized against the first frames, and the analysis segment
% is the 200 frames that follow the 100-frame normalization reference (frames
% 101:300), smoothed with a three-point median filter.
%
% Per animal and state:
%   - signal amplitude: mean normalized intensity over the 22 ROIs;
%   - connectivity: Pearson correlation between the ROI time series;
%   - network deviation: 1 - r between the upper-triangle connectivity profiles
%     of the state and of the animal's baseline segment.
% With one 20-s segment per state, each animal contributes one observation per
% state. Cross-animal averages of the clips are returned for display traces
% only; group connectivity is the average of the per-animal matrices.

if nargin < 1 || isempty(cfg)
    cfg = aergt_preprocess_config_template();
end

state_labels = {'Baseline', 'Seizure', 'Recovery'};
[Mice, state_files] = local_state_recordings(cfg);

fps = cfg.signal.fps;
box_half = cfg.signal.box_half;
fig4_win_len = 200;   % 20 s at 10 Hz
fig4_win_step = 40;   % 4-s stride; the 200-frame segment holds one window
if isfield(cfg, 'fig4') && isfield(cfg.fig4, 'display_roi_index')
    main_roi_idx = cfg.fig4.display_roi_index;
else
    main_roi_idx = 15;
end
deviation_metric = 'corrdist';

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
    roi_map = aergt_map_rois_similarity( ...
        mouse.Coords, ...
        cfg.registration.fixed_points, ...
        cfg.registration.roi_positions, ...
        image_size, ...
        box_half);

    for state_idx = 1:num_states
        file_path = fullfile(mouse.Folder, state_files{mouse_idx, state_idx});
        if ~exist(file_path, 'file')
            error('compute_fig4_ios_state_metrics:MissingStateFile', ...
                'Missing state file for %s: %s', mouse.Name, file_path);
        end

        video = aergt_read_tiff_stack(file_path, 300);
        if size(video, 3) < 300
            error('compute_fig4_ios_state_metrics:ShortRecording', ...
                'The analysis needs at least 300 frames: %s.', file_path);
        end

        roi_traces = aergt_extract_roi_timeseries(video, roi_map, box_half);
        clear video;

        normalized_traces = aergt_normalize_roi_traces(roi_traces, cfg.signal.baseline_frames);
        clip = medfilt1(normalized_traces(101:300, :), 3, [], 1);

        mouse_clips{mouse_idx, state_idx} = clip;
        mouse_state_trace{mouse_idx, state_idx} = mean(clip, 2, 'omitnan');
        mouse_display_trace{mouse_idx, state_idx} = mean(clip(:, main_roi_idx), 2, 'omitnan');

        % Connectivity of the animal over the whole (single 20-s) segment
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

% Cross-animal averaged clips are used for display traces only; group
% connectivity is the average of the per-animal matrices.
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

    adjacency = mean(cat(3, mouse_adjacency{:, state_idx}), 3, 'omitnan');
    adjacency(1:size(adjacency, 1)+1:end) = 1;
    adjacency_matrices{state_idx} = adjacency;
end

% Network deviation at the animal level: one value per animal and state.
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
    'deviation_unit', 'per_animal');
end

function [Mice, state_files] = local_state_recordings(cfg)
% One baseline, seizure, and recovery recording per animal from the Fig. 4
% table; folders and registration landmarks come from the animal table.
required = ["animal", "baseline_file", "seizure_file", "recovery_file"];
T = aergt_read_spec_table(cfg.spec.fig4_state_recordings);
missing = required(~ismember(required, string(T.Properties.VariableNames)));
if ~isempty(missing)
    error('compute_fig4_ios_state_metrics:MissingColumns', ...
        '%s lacks the column(s) %s.', cfg.spec.fig4_state_recordings, strjoin(missing, ', '));
end
Animals = aergt_read_animal_table(cfg, 'labeled');

names = strtrim(string(T.animal));
files = [strtrim(string(T.baseline_file)), strtrim(string(T.seizure_file)), strtrim(string(T.recovery_file))];
if any(names == "") || any(startsWith(names, "ANIMAL_ID")) || ...
        any(files == "") || any(contains(files, "RECORDING"))
    error('compute_fig4_ios_state_metrics:Placeholder', ...
        '%s still contains placeholder rows.', cfg.spec.fig4_state_recordings);
end

Mice = repmat(Animals(1), 1, height(T));
for k = 1:height(T)
    a = find(strcmpi({Animals.Name}, names(k)), 1);
    if isempty(a)
        error('compute_fig4_ios_state_metrics:UnknownAnimal', ...
            'Animal %s is not listed in %s.', names(k), cfg.spec.animals);
    end
    Mice(k) = Animals(a);
end
state_files = cell(height(T), 3);
for state_idx = 1:3
    state_files(:, state_idx) = cellstr(files(:, state_idx));
end
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
