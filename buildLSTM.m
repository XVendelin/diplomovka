function lgraph = buildLSTM(input_size)

% Create layer graph for complex architecture with skip connections
lgraph = layerGraph();

%% INPUT BRANCH
lgraph = addLayers(lgraph, [
    sequenceInputLayer(input_size, 'Name', 'input')
    
    % Dense preprocessing block
    fullyConnectedLayer(256, 'Name', 'fc_pre1')
    reluLayer('Name', 'relu_pre1')
    dropoutLayer(0.1, 'Name', 'drop_pre1')
    
    fullyConnectedLayer(256, 'Name', 'fc_pre2')
    reluLayer('Name', 'relu_pre2')
    batchNormalizationLayer('Name', 'bn_pre')
]);

%% FIRST BIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    bilstmLayer(256, 'OutputMode', 'sequence', 'Name', 'bilstm1')  % 256*2=512 output
    batchNormalizationLayer('Name', 'bn1')
    dropoutLayer(0.2, 'Name', 'drop1')
]);

%% ATTENTION BLOCK 1
lgraph = addLayers(lgraph, [
    selfAttentionLayer(8, 64, 'Name', 'attention1')  % 8 heads, 64-dim
    batchNormalizationLayer('Name', 'bn_att1')
    dropoutLayer(0.2, 'Name', 'drop_att1')
]);

%% SECOND BIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    bilstmLayer(256, 'OutputMode', 'sequence', 'Name', 'bilstm2')  % 256*2=512 output
    batchNormalizationLayer('Name', 'bn2')
    dropoutLayer(0.25, 'Name', 'drop2')
]);

%% RESIDUAL CONNECTION 1 (skip from drop1 to add1)
% Both drop1 and drop2 output 512 dimensions (BiLSTM 256*2)
lgraph = addLayers(lgraph, [
    additionLayer(2, 'Name', 'add1')
]);

%% ATTENTION BLOCK 2
lgraph = addLayers(lgraph, [
    selfAttentionLayer(8, 64, 'Name', 'attention2')
    batchNormalizationLayer('Name', 'bn_att2')
    dropoutLayer(0.2, 'Name', 'drop_att2')
]);

%% THIRD BIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    bilstmLayer(192, 'OutputMode', 'sequence', 'Name', 'bilstm3')  % 192*2=384 output
    batchNormalizationLayer('Name', 'bn3')
    dropoutLayer(0.3, 'Name', 'drop3')
]);

%% FOURTH UNIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    lstmLayer(256, 'OutputMode', 'sequence', 'Name', 'lstm4')
    batchNormalizationLayer('Name', 'bn4')
    dropoutLayer(0.3, 'Name', 'drop4')
]);

%% Projection layer for second residual (384->256)
lgraph = addLayers(lgraph, [
    fullyConnectedLayer(256, 'Name', 'fc_proj2')  % Project 384->256
    reluLayer('Name', 'relu_proj2')
]);

%% RESIDUAL CONNECTION 2
lgraph = addLayers(lgraph, [
    additionLayer(2, 'Name', 'add2')
]);

%% FIFTH UNIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'lstm5')
    batchNormalizationLayer('Name', 'bn5')
    dropoutLayer(0.3, 'Name', 'drop5')
]);

%% ATTENTION BLOCK 3
lgraph = addLayers(lgraph, [
    selfAttentionLayer(4, 32, 'Name', 'attention3')  % 4 heads, 32-dim
    batchNormalizationLayer('Name', 'bn_att3')
    dropoutLayer(0.2, 'Name', 'drop_att3')
]);

%% SIXTH UNIDIRECTIONAL LSTM BLOCK
lgraph = addLayers(lgraph, [
    lstmLayer(64, 'OutputMode', 'sequence', 'Name', 'lstm6')
    batchNormalizationLayer('Name', 'bn6')
    dropoutLayer(0.25, 'Name', 'drop6')
]);

%% DENSE OUTPUT BLOCK
lgraph = addLayers(lgraph, [
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop_fc1')
    
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.3, 'Name', 'drop_fc2')
    
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.2, 'Name', 'drop_fc3')
    
    fullyConnectedLayer(4, 'Name', 'fc_out')
    regressionLayer('Name', 'output')
]);

%% CONNECT ALL LAYERS
% Main path
lgraph = connectLayers(lgraph, 'bn_pre', 'bilstm1');
lgraph = connectLayers(lgraph, 'drop1', 'attention1');
lgraph = connectLayers(lgraph, 'drop_att1', 'bilstm2');
lgraph = connectLayers(lgraph, 'drop2', 'add1/in1');

% Skip connection 1 (drop1 -> add1) - both are 512-dim
lgraph = connectLayers(lgraph, 'drop1', 'add1/in2');

% Continue main path
lgraph = connectLayers(lgraph, 'add1', 'attention2');
lgraph = connectLayers(lgraph, 'drop_att2', 'bilstm3');
lgraph = connectLayers(lgraph, 'drop3', 'lstm4');
lgraph = connectLayers(lgraph, 'drop4', 'add2/in1');

% Skip connection 2 with projection (drop3 is 384-dim, project to 256)
lgraph = connectLayers(lgraph, 'drop3', 'fc_proj2');
lgraph = connectLayers(lgraph, 'relu_proj2', 'add2/in2');

% Continue to output
lgraph = connectLayers(lgraph, 'add2', 'lstm5');
lgraph = connectLayers(lgraph, 'drop5', 'attention3');
lgraph = connectLayers(lgraph, 'drop_att3', 'lstm6');
lgraph = connectLayers(lgraph, 'drop6', 'fc1');

% Visualize the architecture
% figure('Name', 'Network Architecture');
% plot(lgraph);
% title('Ultra-Complex LSTM with Residual Connections');

% Analyze network
% fprintf('\n=== Network Analysis ===\n');
% analyzeNetwork(lgraph);

end