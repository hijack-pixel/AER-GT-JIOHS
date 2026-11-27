function T = aergt_read_spec_table(path)
%AERGT_READ_SPEC_TABLE Read a specification CSV as a table.
%
% Text columns are returned as strings and numeric columns as doubles (empty
% cells become NaN).

if ~exist(path, 'file')
    error('aergt_read_spec_table:MissingFile', ...
        'Specification table not found: %s', path);
end

T = readtable(path, 'TextType', 'string', 'Delimiter', ',');
end
