function report = validate_aert_sample_folder(folder_path, is_labeled)
%VALIDATE_AERT_SAMPLE_FOLDER Validate processed AER-GT MATLAB graph samples.
%
% report = validate_aert_sample_folder(folder_path, is_labeled)
%
% Required sample schema:
%   AERT_Sample.NodeFeats: 22 x 6
%   AERT_Sample.Adjacency: 22 x 22
%   AERT_Sample.MouseID: subject identifier
%   AERT_Sample.Label: 0/1/2 for labeled samples, -1 for pretraining controls

if nargin < 2
    is_labeled = true;
end

files = dir(fullfile(folder_path, '*.mat'));
if isempty(files)
    error('No .mat files found in: %s', folder_path);
end

num_valid = 0;
num_invalid = 0;
labels = [];
mouse_ids = strings(0);
invalid_files = strings(0);

for i = 1:numel(files)
    sample_path = fullfile(files(i).folder, files(i).name);
    try
        data = load(sample_path, 'AERT_Sample');
        if ~isfield(data, 'AERT_Sample')
            error('Missing AERT_Sample variable.');
        end
        sample = data.AERT_Sample;
        if ~isfield(sample, 'NodeFeats') || ~isequal(size(sample.NodeFeats), [22, 6])
            error('NodeFeats must be 22 x 6.');
        end
        if ~isfield(sample, 'Adjacency') || ~isequal(size(sample.Adjacency), [22, 22])
            error('Adjacency must be 22 x 22.');
        end
        if ~isfield(sample, 'MouseID')
            error('Missing MouseID.');
        end
        if ~isfield(sample, 'Label')
            error('Missing Label.');
        end
        label_value = double(sample.Label);
        if is_labeled && ~ismember(label_value, [0, 1, 2])
            error('Labeled samples must use labels 0, 1, or 2.');
        end
        if ~is_labeled && label_value ~= -1
            error('Pretraining control samples should use label -1.');
        end
        num_valid = num_valid + 1;
        labels(end + 1) = label_value; %#ok<AGROW>
        mouse_ids(end + 1) = string(sample.MouseID); %#ok<AGROW>
    catch ME
        num_invalid = num_invalid + 1;
        invalid_files(end + 1) = string(sample_path + " :: " + ME.message); %#ok<AGROW>
    end
end

report = struct();
report.folder = folder_path;
report.num_files = numel(files);
report.num_valid = num_valid;
report.num_invalid = num_invalid;
report.unique_mouse_ids = unique(mouse_ids);
report.unique_labels = unique(labels);
report.invalid_files = invalid_files;

disp(report);
end

