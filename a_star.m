clc; clear; close all;

coords = [280 400; 280 520; 
          400 520; 400 400];
map = druhy('image.jpg', coords);

nmap = map; 
nmap(nmap ~= 1) = 0;

waypoints = [10 20;
            100 10;
            100 18;
            10 30;
            10 40;
            100 28;
            100 40;
            10 50;
            10 65;
            100 55;
            100 65;
            10 75;
            10 85;
            100 75;
            100 85;
            10 95;
            10 103;
            100 95;
            10 size(map,2)-10;
            size(map,1)-20, size(map,2)-15];

fullPath = [];
for i = 1:(size(waypoints,1)-1)
    start = waypoints(i,:);
    goal = waypoints(i+1,:);
    segment = astar_normal(map, start, goal);

    if isempty(segment)
        disp(['No path found between waypoint ', num2str(i), ' and ', num2str(i+1)]);
        break;
    end

    if i > 1
        segment = segment(2:end,:);
    end

    fullPath = [fullPath; segment];
end

figure;
imshow(nmap, []); 
hold on;
plot(fullPath(:,2), fullPath(:,1), 'r-', 'LineWidth', 2);
plot(waypoints(:,2), waypoints(:,1), 'go', 'MarkerSize',10,'MarkerFaceColor','g');
title('A* Path');


function path = astar_normal(map, start, goal)

    [rows, cols] = size(map);

    if map(start(1), start(2)) == 1 || map(goal(1), goal(2)) == 1
        path = [];
        return;
    end

    openSet = false(rows, cols);
    cameFrom = zeros(rows, cols, 2);
    gScore = inf(rows, cols);
    fScore = inf(rows, cols);

    gScore(start(1), start(2)) = 0;
    fScore(start(1), start(2)) = heuristic(start, goal);
    openSet(start(1), start(2)) = true;

    while any(openSet(:))
        maskedFS = fScore;
        maskedFS(~openSet) = inf;
        [~, idx] = min(maskedFS(:));
        [cr, cc] = ind2sub(size(map), idx);
        current = [cr, cc];

        if all(current == goal)
            path = current;
            while any(cameFrom(path(1,1), path(1,2),:))
                prev = squeeze(cameFrom(path(1,1), path(1,2),:))';
                path = [prev; path];
            end
            return;
        end

        openSet(cr, cc) = false;

        for dr = -1:1
            for dc = -1:1
                if dr == 0 && dc == 0, continue; end
                nr = cr + dr; nc = cc + dc;
                if nr < 1 || nr > rows || nc < 1 || nc > cols, continue; end

                if map(nr, nc) == 1, continue; end

                cost = sqrt(dr^2 + dc^2);
                tentative = gScore(cr, cc) + cost;

                if tentative < gScore(nr, nc)
                    cameFrom(nr,nc,:) = current;
                    gScore(nr,nc) = tentative;
                    fScore(nr,nc) = tentative + heuristic([nr,nc], goal);
                    openSet(nr,nc) = true;
                end
            end
        end
    end

    path = [];
end


function h = heuristic(p, goal)
    h = abs(p(1) - goal(1)) + abs(p(2) - goal(2));
end
