function aergt_write_control_quality_report(output_dir, mouse_id, roi_traces, normalized_traces, fps, roi_labels)
%AERGT_WRITE_CONTROL_QUALITY_REPORT Optionally write a per-control quality figure.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fig = figure( ...
    'Name', mouse_id, ...
    'Color', 'w', ...
    'Position', [50, 50, 1400, 900], ...
    'Visible', 'off' ...
);

subplot(2, 2, 1);
plot(roi_traces, 'LineWidth', 0.5);
title([mouse_id ' - raw ROI intensity']);
xlabel('Frames');
ylabel('Intensity');
grid on;

subplot(2, 2, 2);
plot(normalized_traces, 'LineWidth', 0.5);
title('Percent intensity change');
xlabel('Frames');
ylabel('Change (%)');
grid on;

subplot(2, 2, 3);
[pxx, f_axis] = pwelch(normalized_traces(:, 1), 512, 256, 512, fps);
loglog(f_axis, pxx, 'LineWidth', 1.5, 'Color', 'r');
xline(0.1, '--', '0.1 Hz', 'LabelVerticalAlignment', 'bottom');
title(['PSD fingerprint (' roi_labels{1} ')']);
xlabel('Frequency (Hz)');
ylabel('Power');
xlim([0.01, 1]);
grid on;

subplot(2, 2, 4);
conn = corr(normalized_traces);
conn(isnan(conn)) = 0;
imagesc(conn);
colormap(jet);
caxis([0.5, 1]);
colorbar;
set(gca, 'XTick', 1:22, 'XTickLabel', roi_labels, 'YTick', 1:22, 'YTickLabel', roi_labels);
xtickangle(45);
title('Connectivity matrix');

saveas(fig, fullfile(output_dir, [aergt_safe_filename(mouse_id) '_Quality_Report.png']));
close(fig);
end
