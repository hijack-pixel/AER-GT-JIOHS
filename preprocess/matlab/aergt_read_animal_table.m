function Animals = aergt_read_animal_table(cfg, cohort)
%AERGT_READ_ANIMAL_TABLE Animals of one cohort from the specification table.
%
%   Animals = aergt_read_animal_table(cfg, cohort)
%
% cfg.spec.animals is a CSV with one row per animal and the columns
%   cohort, animal, folder, landmark1_x, landmark1_y, landmark2_x, landmark2_y
% ``folder`` is the animal's raw folder, relative to the root of the cohort (or
% absolute); the two landmarks are the animal's registration landmarks in pixel
% coordinates (the points mapped onto cfg.registration.fixed_points).
%
% Animals is a struct array with the fields Name, Folder, and Coords.

required = ["cohort", "animal", "folder", "landmark1_x", "landmark1_y", "landmark2_x", "landmark2_y"];
T = aergt_read_spec_table(cfg.spec.animals);
missing = required(~ismember(required, string(T.Properties.VariableNames)));
if ~isempty(missing)
    error('aergt_read_animal_table:MissingColumns', ...
        '%s lacks the column(s) %s.', cfg.spec.animals, strjoin(missing, ', '));
end

T = T(strtrim(string(T.cohort)) == string(cohort), :);
if isempty(T)
    error('aergt_read_animal_table:NoAnimals', ...
        'No %s animals are listed in %s.', cohort, cfg.spec.animals);
end

root = aergt_cohort_root(cfg, cohort);
names = strtrim(string(T.animal));
folders = strtrim(string(T.folder));
if any(names == "") || any(folders == "")
    error('aergt_read_animal_table:EmptyEntry', ...
        'Every row of %s needs an animal identifier and a folder.', cfg.spec.animals);
end
if numel(unique(names)) ~= numel(names)
    error('aergt_read_animal_table:DuplicateAnimal', ...
        'Animal identifiers must be unique within a cohort.');
end

Animals = struct('Name', {}, 'Folder', {}, 'Coords', {});
for k = 1:height(T)
    coords = [T.landmark1_x(k), T.landmark1_y(k); T.landmark2_x(k), T.landmark2_y(k)];
    if any(~isfinite(coords(:)))
        error('aergt_read_animal_table:MissingLandmarks', ...
            'Missing registration landmarks for animal %s.', names(k));
    end
    folder = char(folders(k));
    if folder(1) ~= filesep && ~(length(folder) > 1 && folder(2) == ':')
        folder = fullfile(root, folder);
    end
    Animals(end + 1) = struct('Name', char(names(k)), 'Folder', folder, 'Coords', coords); %#ok<AGROW>
end
end
