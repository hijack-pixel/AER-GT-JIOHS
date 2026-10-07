function AERT_Sample = processed_sample_schema()
%PROCESSED_SAMPLE_SCHEMA Return an empty AERT_Sample schema example.
%
% This helper documents the processed graph-sample structure expected by the
% Python training code. It is not a replacement for raw IOS preprocessing.

AERT_Sample = struct();
AERT_Sample.NodeFeats = zeros(22, 6, 'single');
AERT_Sample.Adjacency = eye(22, 'single');
AERT_Sample.Label = int32(-1);
AERT_Sample.MouseID = '';
AERT_Sample.OriginalFile = '';
AERT_Sample.WindowIndex = int32(-1);
AERT_Sample.WindowStartFrame = int32(-1);
AERT_Sample.WindowEndFrame = int32(-1);
AERT_Sample.FPS = int32(10);
AERT_Sample.WinSec = int32(20);
AERT_Sample.StepSec = int32(4);
end

