close all;

plot_width      = 600;                          % ensures able to see all plots easily
plot_height     = 350;                          % ensures able to see all plots easily

%% Visualize optimal vs IC decoy motion
figure(4); hold on; sgtitle('Optimal vs. Initial (Defender Motions)');
set(gcf, 'Color', 'w');  % Set the figure background to white
% pcon = plot(PolyCon(:,1), PolyCon(:,2), 'g-', 'LineWidth', 2);  % Green outline of the polygon safe-zone
pcon = fill(PolyCon(:,1), PolyCon(:,2), [0.75, 1, 0.75], 'EdgeColor', 'g', 'LineWidth', 2, 'FaceAlpha', 0.2);  % Light green filled area with green edge
if modelname == "CNmcFull"
    arrow_length = 1.5; % Desired arrow length (0.5 for 20, 1.5 for Full)
else
    arrow_length = 0.5;
end
arrow_head   = 1; % doesn't do anything
%% Define custom colors (10 distinct colors)
colors = [
    0.1216, 0.4667, 0.7059;  % Blue
    1.0000, 0.4980, 0.0549;  % Orange
    0.1725, 0.6275, 0.1725;  % Green
    0.8392, 0.1529, 0.1569;  % Red
    0.5804, 0.4039, 0.7412;  % Purple
    0.5490, 0.3373, 0.2941;  % Brown
    0.8500, 0.3250, 0.0980;  % Dark Orange
    0.4980, 0.4980, 0.4980;  % Gray
    0.5020, 0,      0;       % Maroon
    0.0902, 0.7451, 0.8118;  % Cyan
];

% Plot both initial and optimal motions with arrowheads
for i = 1:size(DefendersPos_x, 2)
    % Choose the color from the defined colors array (cycle through if necessary)
    colorD = colors(mod(i-1, 10) + 1, :);  % Mod function cycles the index between 1 and 10

    % Plot the initial decoy motion (goD)
    goD(i) = plot(DefendersPos_x(:,i), DefendersPos_y(:,i), '--', 'Color', colorD, 'LineWidth', 1);
    
    % Add arrowheads for the initial motion
    U = DefendersPos_x(end, i) - DefendersPos_x(end-1, i);
    V = DefendersPos_y(end, i) - DefendersPos_y(end-1, i);
    magnitude = sqrt(U^2 + V^2);
    if magnitude ~= 0
        U_normalized = (U / magnitude) * arrow_length;
        V_normalized = (V / magnitude) * arrow_length;
    else
        U_normalized = 0;
        V_normalized = 0;
    end
    quiver(DefendersPos_x(end, i), DefendersPos_y(end, i), ...
           U_normalized, V_normalized, ...
           'Color', colorD, 'MaxHeadSize', arrow_head, 'LineWidth', 1, ...
           'AutoScale', 'off', 'Alignment', 'head');
    
    % Plot the optimal motion (goT)
    goT(i) = plot(Px(:,i), Py(:,i), 'Color', colorD, 'LineWidth', 2);
    % plot(Px(end,i), Py(end,i), '^', 'MarkerSize',8,'MarkerFaceColor',colorD,'Color', colorD, 'LineWidth', 2);
    
    % % Add arrowheads for the optimal motion
    % U = Px(end, i) - Px(end-1, i);
    % V = Py(end, i) - Py(end-1, i);
    % magnitude = sqrt(U^2 + V^2);
    % if magnitude ~= 0
    %     U_normalized = (U / magnitude) * arrow_length;
    %     V_normalized = (V / magnitude) * arrow_length;
    % else
    %     U_normalized = 0;
    %     V_normalized = 0;
    % end
    % quiver(Px(end, i), Py(end, i), ...
    %        U_normalized, V_normalized, ...
    %        'Color', colorD, 'MaxHeadSize', arrow_head, 'LineWidth', 2, ...
    %        'AutoScale', 'off', 'Alignment', 'head');
end

hvu = scatter(0, 0, 25^2, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5); % HVU is yellow
text(0, 0, 'HVU', 'VerticalAlignment','middle', 'HorizontalAlignment','center','FontSize', 10);  % Display HVU label
axis equal;       % Force the plot to have equal scaling for both axes

% Create the legend using the custom lines
blackSolidLine = plot(nan, nan, 'k-', 'LineWidth', 2);   % Black solid line for Optimal Motion
blackDashLine = plot(nan, nan, 'k--', 'LineWidth', 1);   % Black dashed line for Initial Motion
legend([blackSolidLine, blackDashLine, pcon], 'Optimal Motion', 'Initial Motion', 'Allowable OpArea', 'Location', 'best');

axis off;
hold off;
set(gcf, 'Position', [700, 500, plot_width, plot_height]); % [left, bottom, width, height]
