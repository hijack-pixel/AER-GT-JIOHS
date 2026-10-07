function Mice = aergt_build_paper_mouse_table(cfg)
%AERGT_BUILD_PAPER_MOUSE_TABLE Return the five epilepsy mice used in paper training.

root_dir = cfg.raw.epilepsy_root;

Mice = struct('Name', {}, 'Folder', {}, 'Coords', {});

Mice(1).Name = 'Mouse 1';
Mice(1).Folder = fullfile(root_dir, 'mouse1', 'recording');
Mice(1).Coords = [64, 48; 64, 110];

Mice(2).Name = 'Mouse 2';
Mice(2).Folder = fullfile(root_dir, 'mouse2', 'recording');
Mice(2).Coords = [61, 37; 60, 92];

Mice(3).Name = 'Mouse 3';
Mice(3).Folder = fullfile(root_dir, 'mouse3', 'recording');
Mice(3).Coords = [67, 56; 67, 108];

Mice(4).Name = 'Mouse 4';
Mice(4).Folder = fullfile(root_dir, 'mouse4', 'recording');
Mice(4).Coords = [67, 48; 67, 115];

Mice(5).Name = 'Mouse 5';
Mice(5).Folder = fullfile(root_dir, 'mouse5', 'recording');
Mice(5).Coords = [65, 48; 66, 107];
end
