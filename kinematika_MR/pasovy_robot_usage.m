state = [0; 0; 0; 0; 0; 0.2; 0];
dt = 0.01;
steps = 1000;
M = [5; 3; 5; 3];  

figure;
h_plot = plot(state(1), state(2), 'b-'); 
hold on; 
h_robot = plot(state(1), state(2), 'ro', 'MarkerFaceColor', 'r');
grid on;
xlim([-3.5, 3.5]);
ylim([-3, 0.5]);

for k = 1:steps
    state = trackedRobotDynamics(state, M, dt);
    
    x_current = state(1);
    y_current = state(2);

    
    XData = [get(h_plot, 'XData'), x_current];
    YData = [get(h_plot, 'YData'), y_current];
    
    set(h_plot, 'XData', XData, 'YData', YData);

    set(h_robot, 'XData', x_current, 'YData', y_current);

    drawnow;
    
end