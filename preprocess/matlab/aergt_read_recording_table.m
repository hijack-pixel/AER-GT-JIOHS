function Recordings = aergt_read_recording_table(cfg, cohort, Animals)
%AERGT_READ_RECORDING_TABLE Recordings of one cohort from the specification table.
%
%   Recordings = aergt_read_recording_table(cfg, cohort, Animals)
%
% cfg.spec.recordings is a CSV with the columns
%   cohort, animal, recording, file, order, label
%
% A labeled recording is a single file whose label is baseline, seizure, or
% recovery. An unlabeled recording (pretraining or control cohort) may consist
% of several TIFF chunks that share the ``recording`` identifier and are
% concatenated in ``order``; its label is -1.
%
% Recordings is a struct array ordered by animal (as in Animals) and then by
% first appearance, with the fields Animal, Recording, Files, and Label.

required = ["cohort", "animal", "file"];
T = aergt_read_spec_table(cfg.spec.recordings);
columns = string(T.Properties.VariableNames);
missing = required(~ismember(required, columns));
if ~isempty(missing)
    error('aergt_read_recording_table:MissingColumns', ...
        '%s lacks the column(s) %s.', cfg.spec.recordings, strjoin(missing, ', '));
end
if ismember("recording", columns)
    recording_ids = strtrim(string(T.recording));
else
    recording_ids = repmat("", height(T), 1);
end
if ismember("order", columns)
    order = double(T.order);
    order(isnan(order)) = 1;
else
    order = ones(height(T), 1);
end
if ismember("label", columns)
    labels = lower(strtrim(string(T.label)));
else
    labels = repmat("", height(T), 1);
end

keep = strtrim(string(T.cohort)) == string(cohort);
animals = strtrim(string(T.animal(keep)));
recording_ids = recording_ids(keep);
files = strtrim(string(T.file(keep)));
order = order(keep);
labels = labels(keep);
if isempty(animals)
    error('aergt_read_recording_table:NoRecordings', ...
        'No %s recordings are listed in %s.', cohort, cfg.spec.recordings);
end
if any(files == "")
    error('aergt_read_recording_table:EmptyEntry', ...
        'Every row of %s needs a TIFF file name.', cfg.spec.recordings);
end
known = string({Animals.Name});
listed = unique(animals);
if any(~ismember(listed, known))
    orphan = listed(~ismember(listed, known));
    error('aergt_read_recording_table:UnknownAnimal', ...
        'Recordings are listed for animals missing from the animal table: %s', strjoin(orphan, ', '));
end

entries = cell(1, 0);
for a = 1:numel(Animals)
    in_animal = animals == known(a);
    ids = unique(recording_ids(in_animal), 'stable');
    for r = 1:numel(ids)
        idx = find(in_animal & recording_ids == ids(r));
        [~, order_idx] = sort(order(idx));
        idx = idx(order_idx);
        if strcmpi(cohort, 'labeled')
            label_id = local_label_id(labels(idx(1)), cfg.spec.recordings);
            if numel(idx) > 1
                error('aergt_read_recording_table:ChunkedLabeledRecording', ...
                    'A labeled recording must be a single file (%s, %s).', Animals(a).Name, ids(r));
            end
        else
            label_id = -1;
        end
        % the file list is assigned after struct(): a cell argument would expand
        % the struct into one element per file
        entry = struct('Animal', Animals(a).Name, 'Recording', char(ids(r)), 'Label', label_id);
        entry.Files = cellstr(files(idx));
        entries{end + 1} = entry; %#ok<AGROW>
    end
end
Recordings = [entries{:}];
if isempty(entries)
    Recordings = struct('Animal', {}, 'Recording', {}, 'Files', {}, 'Label', {});
end
end

function label_id = local_label_id(label, spec_path)
switch label
    case "baseline"
        label_id = 0;
    case "seizure"
        label_id = 1;
    case "recovery"
        label_id = 2;
    otherwise
        error('aergt_read_recording_table:UnknownLabel', ...
            'Label "%s" in %s is not baseline, seizure, or recovery.', label, spec_path);
end
end
