function specs = aergt_control_specs()
%AERGT_CONTROL_SPECS Return CON1 and CON2 saline-control specifications.

specs = struct('Name', {}, 'NumMice', {}, 'MousePoints', {}, 'MouseIdPattern', {}, 'FilenameMode', {});

specs(1).Name = 'CON1';
specs(1).NumMice = 10;
specs(1).MouseIdPattern = 'CON_%d';
specs(1).FilenameMode = 'con1';
specs(1).MousePoints = [
    122, 100, 126, 228;
    122, 112, 132, 226;
    126, 88, 130, 212;
    118, 126, 120, 228;
    122, 80, 134, 222;
    121, 71, 124, 197;
    114, 88, 120, 220;
    118, 75, 120, 207;
    134, 68, 137, 174;
    126, 80, 122, 220
];

specs(2).Name = 'CON2';
specs(2).NumMice = 9;
specs(2).MouseIdPattern = 'CON2_Mouse%02d';
specs(2).FilenameMode = 'con2';
specs(2).MousePoints = [
    124, 61, 128, 199;
    126, 66, 138, 182;
    118, 108, 118, 220;
    125, 79, 125, 214;
    111, 22, 115, 167;
    121, 72, 126, 207;
    126, 57, 132, 201;
    124, 89, 134, 225;
    123, 89, 126, 214
];
end
