function clean_name = aergt_safe_filename(raw_name)
%AERGT_SAFE_FILENAME Convert a source name into a portable file stem.

clean_name = regexprep(raw_name, '[^A-Za-z0-9_-]+', '_');
clean_name = regexprep(clean_name, '_+', '_');
clean_name = regexprep(clean_name, '^_|_$', '');
end
