window = 50;

foldername = 'savedAgents_TD3_neviem_2_2';
files = dir(fullfile(foldername, 'Agent*.mat'));

if ~isempty(files)
    [~, idx] = sort([files.datenum], 'descend');
    newestFile = fullfile(files(idx(1)).folder, files(idx(1)).name);
    
    data = load(newestFile);
    rewards = data.savedAgentResult.EpisodeReward;
else
    error('No agent files found in the directory.');
end

load("td3_neviem.mat");
data_A = trainingStats.EpisodeReward; 
load("td3_neviem2.mat");
data_B = [trainingStats.EpisodeReward; rewards]; 

x_A = 1:length(data_A);
x_B = 1:length(data_B);

[m_A, p10_A, p90_A] = deal(zeros(size(data_A)));
[m_B, p10_B, p90_B] = deal(zeros(size(data_B)));

for i = 1:length(data_A)
    idx = max(1, i-window):i;
    m_A(i)   = mean(data_A(idx));
    p10_A(i) = prctile(data_A(idx), 10);
    p90_A(i) = prctile(data_A(idx), 90);
end

for i = 1:length(data_B)
    idx = max(1, i-window):i;
    m_B(i)   = mean(data_B(idx));
    p10_B(i) = prctile(data_B(idx), 10);
    p90_B(i) = prctile(data_B(idx), 90);
end

figure; hold on;

fill([x_A, fliplr(x_A)], [p10_A', fliplr(p90_A')], [0.2 0.6 1], 'EdgeColor', 'none', 'FaceAlpha', 0.2, 'HandleVisibility', 'off');
plot(x_A, m_A, 'Color', [0 0.4 0.8], 'LineWidth', 2, 'DisplayName', 'Prvý beh');

fill([x_B, fliplr(x_B)], [p10_B', fliplr(p90_B')], [1 0.4 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.2, 'HandleVisibility', 'off');
plot(x_B, m_B, 'Color', [0.8 0 0], 'LineWidth', 2, 'DisplayName', 'Druhý beh');

grid on;
xlabel('Epizóda');
ylabel('Odmena');
legend('Location', 'best');
