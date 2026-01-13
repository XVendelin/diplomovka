%% ========== DATA COLLECTION ==========
fprintf('\nCollecting training data...\n');

all_sequences = {};
all_targets = {};
success_count = 0;

for ep = 1:num_episodes
    % Random starting position along early part of path
    start_idx = max(1, min(10, size(pathWorld,1)));
    start_idx = randi([1, start_idx]);
    
    init_pos = pathWorld(start_idx, :)';
    init_theta = randn * 0.2;
    state = [init_pos; init_theta; 0; 0; 0.22; 0];
    
    sequence = [];
    targets = [];
    current_path_idx = start_idx;
    
    try
        for step = 1:max_steps
            x = state(1);
            y = state(2);
            theta = state(3);
            
            % Find nearest point on path and look-ahead target
            [current_path_idx, target_point] = findPathTarget(pathWorld, [x, y], ...
                current_path_idx, path_follow_distance);
            
            % Extract local terrain
            local_terrain = extractLocalTerrain(x, y, map, res, obs_radius, obs_size);
            
            % Calculate features relative to path target
            dx = target_point(1) - x;
            dy = target_point(2) - y;
            dist_to_target = sqrt(dx^2 + dy^2);
            angle_to_target = atan2(dy, dx);
            heading_error = wrapToPi(angle_to_target - theta);
            
            % Distance to final goal
            dist_to_goal = norm(goal_pos - [x; y]);
            
            % Features: [dist_to_target, heading_error, dist_to_goal, v, omega, terrain]
            terrain_vec = reshape(local_terrain, [], 1);
            features = [theta; dist_to_target; heading_error; dist_to_goal; 
                        state(4); state(5); terrain_vec];

            
            % Path-following controller with exploration
            exploitation_rate = 0.8 + 0.2*(ep/num_episodes);
            
            if rand < exploitation_rate
                M = pathFollowingController(state, target_point, local_terrain);
            else
                M = randn(4,1) * 2.5 + 1.5;  % Exploration
            end
            
            M = max(min(M, 10), -10);
            
            % Store data
            sequence = [sequence; features'];
            targets = [targets; M'];
            
            % Simulate
            state_new = trackedRobotDynamicsWithTerrain(state, M, dt, map, res);
            
            % Check if stuck
            if norm(state_new(1:2) - state(1:2)) < 0.001
                break;
            end
            
            state = state_new;
            
            % Success if reached goal
            if dist_to_goal < 0.1
                fprintf('Episode %d: Reached goal in %d steps!\n', ep, step);
                success_count = success_count + 1;
                break;
            end
            
            % Also succeed if made good progress along path
            if current_path_idx >= size(pathWorld,1) - 10
                fprintf('Episode %d: Reached end of path!\n', ep);
                success_count = success_count + 1;
                break;
            end
        end
    catch ME
        fprintf('Episode %d error: %s\n', ep, ME.message);
        continue;
    end
    
    % Accept if made reasonable progress (moved forward on path)
    progress = current_path_idx - start_idx;
    if progress > 20 || dist_to_goal < 0.1
        all_sequences{end+1} = sequence;
        all_targets{end+1} = targets;
    end
    
    if mod(ep, 25) == 0
        fprintf('Episodes: %d/%d | Sequences: %d | Success: %d\n', ...
            ep, num_episodes, length(all_sequences), success_count);
    end
end

fprintf('\n=== Collection Summary ===\n');
fprintf('Sequences: %d | Success: %d\n', length(all_sequences), success_count);

if isempty(all_sequences)
    error('No training data collected!');
end