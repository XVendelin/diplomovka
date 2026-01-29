%% ========== TEST TRAINED AGENT (WITH VIDEO + GIF RECORDING) ==========
load("2_20_2.mat");

agent_cpu = setLearnableParameters( ...
    agent, dlupdate(@gather, getLearnableParameters(agent)));

addpath("kinematika_MR");
fprintf('\n=== Testing Trained Agent ===\n');

test_start = [10; 18] * res;
test_goal  = [100; 10] * res;

state = [test_start; 0; 0; 0; 0; 0];
trajectory = state(1:2)';

max_test_steps = 2000;

%% ---------- VIDEO + GIF SETUP ----------
videoFile = 'agent_test.mp4';
gifFile   = 'agent_test.gif';

vwriter = VideoWriter(videoFile, 'MPEG-4');
vwriter.FrameRate = 10;
open(vwriter);

gifDelay = 1 / vwriter.FrameRate;
isFirstGifFrame = true;
% ---------------------------------------

figure('Position', [100, 100, 1500, 600]);

for step = 1:max_test_steps

    x     = state(1);
    y     = state(2);
    theta = state(3);
    v     = state(4);
    omega = state(5);

    local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
    terrain_vec   = reshape(local_terrain, [], 1);

    goal_dx = test_goal(1) - x;
    goal_dy = test_goal(2) - y;
    goal_heading_error = wrapToPi(atan2(goal_dy, goal_dx) - theta);
    dist_to_goal = hypot(goal_dx, goal_dy);

    obs = [
        sin(theta);
        cos(theta);
        sin(goal_heading_error);
        cos(goal_heading_error);
        dist_to_goal / norm(test_goal - test_start);
        state(4) / 5;
        state(5) / 5;
        terrain_vec
    ];

    % ---- Get action from trained agent ----
    M = getAction(agent, obs);
    M = M{1};
    M = max(min(M(:), 10), -10);

    state = trackedRobotDynamics(state, M, dt);
    trajectory = [trajectory; state(1:2)'];

    %% ---------- VISUALIZATION ----------
    clf;

    subplot(1,4,1);
    imagesc(map); colormap gray; hold on;
    plot(trajectory(:,2)/res, trajectory(:,1)/res, 'g-', 'LineWidth', 2);
    plot(test_start(2)/res, test_start(1)/res, 'go', ...
        'MarkerSize', 8, 'MarkerFaceColor', 'g');
    plot(test_goal(2)/res, test_goal(1)/res, ...
        'r*', 'MarkerSize', 18, 'LineWidth', 2);
    drawTrackedRobot(x, y, theta, 0.25, 0.25, res);
    title(sprintf('Step %d | Goal: %.2f m', step, dist_to_goal));
    axis equal tight; hold off;

    subplot(1,4,2);
    bar(M); ylim([-12 12]); grid on;
    title('Motor Torques'); ylabel('Nm');
    xticklabels({'FL','FR','RL','RR'});

    subplot(1,4,3);
    bar([v omega]); ylim([-5 5]); grid on;
    title('Speeds');
    xticklabels({'Linear','Angular'});
    ylabel('m/s   |   rad/s');

    subplot(1,4,4);
    imagesc(local_terrain, [min(map(:)) max(map(:))]);
    colormap(gca, 'gray'); colorbar;
    title('Local View');
    axis equal tight;

    drawnow;

    %% ---------- WRITE VIDEO + GIF ----------
    frame = getframe(gcf);

    % MP4
    writeVideo(vwriter, frame);

    % GIF
    [im, cm] = rgb2ind(frame2im(frame), 256);
    if isFirstGifFrame
        imwrite(im, cm, gifFile, 'gif', ...
            'LoopCount', inf, 'DelayTime', gifDelay);
        isFirstGifFrame = false;
    else
        imwrite(im, cm, gifFile, 'gif', ...
            'WriteMode', 'append', 'DelayTime', gifDelay);
    end
    %% --------------------------------------

    if dist_to_goal < 0.1
        fprintf('SUCCESS! Goal reached in %d steps (%.2f m)\n', ...
            step, dist_to_goal);
        break;
    end

    if step > 200 && norm(trajectory(end,:) - trajectory(end-50,:)) < 0.5
        fprintf('Stuck at step %d (%.2f m from goal)\n', ...
            step, dist_to_goal);
        break;
    end
end

%% ---------- CLOSE VIDEO ----------
close(vwriter);
fprintf('MP4 saved to: %s\n', videoFile);
fprintf('GIF saved to: %s\n', gifFile);
close all;

%% ========== HELPER FUNCTIONS ==========

function local_map = extractLocalTerrain(x,y, map, res, radius, grid_size)
    half = floor(grid_size/2);
    local_map = zeros(grid_size, grid_size);
    
    for i = 1:grid_size
        for j = 1:grid_size
            wx = x + (j - half - 1) * radius / half;
            wy = y + (i - half - 1) * radius / half;
            
            ix = round(wx/res) + 1;
            iy = round(wy/res) + 1;
            
            if ix >= 1 && ix <= size(map,2) && iy >= 1 && iy <= size(map,1)
                local_map(j,i) = map(ix, iy);
            else
                local_map(j,i) = 1.0;
            end
        end
    end
end

function drawTrackedRobot(x, y, theta, L, y_offset, res)

    % Robot corners (rectangle)
    corners_robot = [
        -L, -y_offset;
         L, -y_offset;
         L,  y_offset;
        -L,  y_offset
    ];

    % Rotation matrix
    R = [cos(theta) -sin(theta);
         sin(theta)  cos(theta)];

    % Transform corners to world coordinates
    corners_world = (R * corners_robot')';
    corners_world(:,1) = corners_world(:,1) + x;
    corners_world(:,2) = corners_world(:,2) + y;

    % World → map indices
    col = corners_world(:,2) / res;   % Y → column
    row = corners_world(:,1) / res;   % X → row

    % Draw filled rectangle
    patch( ...
        col, row, 'r', ...
        'EdgeColor', 'r', ...
        'LineWidth', 0.1);

    % --- Draw center-to-front line ---
    front_x = x + (L) * cos(theta);  % front point X
    front_y = y + (L) * sin(theta);  % front point Y

    line([y, front_y]/res, [x, front_x]/res, 'Color', 'k', 'LineWidth', 1);  % black line
end


