%% ========== WEIGHT TRANSFER ==========
fprintf('Transferring weights from SAC to TD3 by Layer Name...\n');

oldData = load('savedAgents_uznefunguje\Agent939.mat');
oldAgent = oldData.saved_agent;

sacActor = getActor(oldAgent); 
sacDlNet = getModel(sacActor);

td3DlNet = getModel(actor);    

td3LayerNames = {td3DlNet.Layers.Name};

for i = 1:numel(td3LayerNames)
    layerName = td3LayerNames{i};
    
    if any(strcmp({sacDlNet.Layers.Name}, layerName))
        try
            params = getLearnableParameters(sacActor, layerName);
            
            if ~isempty(params)
                currentParams = getLearnableParameters(actor, layerName);
                if isequal(size(params{1}), size(currentParams{1}))
                    fprintf('Mapping Layer: %-15s [Success]\n', layerName);
                    actor = setLearnableParameters(actor, layerName, params);
                else
                    fprintf('Skipping Layer: %-15s [Dimension Mismatch]\n', layerName);
                end
            end
        catch
            continue; 
        end
    else
        fprintf('Ignoring Layer: %-15s [Not in SAC]\n', layerName);
    end
end


targetFinalLayer = 'mean_fc2';

if any(strcmp(td3LayerNames, targetFinalLayer)) && ~any(strcmp({sacDlNet.Layers.Name}, targetFinalLayer))
    try
        fprintf('Mapping Final Head: SAC "mean_fc2" -> TD3 "%s"...\n', targetFinalLayer);
        meanParams = getLearnableParameters(sacActor, 'mean_fc2');
        actor = setLearnableParameters(actor, targetFinalLayer, meanParams);
    catch
        fprintf('Final head mapping failed.\n');
    end
end

fprintf('Actor weight transfer complete.\n\n');

%% ========== CRITIC WEIGHT TRANSFER ==========
fprintf('Transferring weights from SAC Critics to TD3 Critics...\n');

sacCritics = getCritic(oldAgent);
td3Critics = {critic1, critic2};

for c = 1:numel(td3Critics)
    targetCritic = td3Critics{c};
    sacCritDlNet = getModel(sacCritics(c)); 
    td3CritDlNet = getModel(targetCritic);
    
    td3CritLayerNames = {td3CritDlNet.Layers.Name};
    fprintf('Updating TD3 Critic %d...\n', c);
    
    for i = 1:numel(td3CritLayerNames)
        layerName = td3CritLayerNames{i};
        
        if any(strcmp({sacCritDlNet.Layers.Name}, layerName))
            try
                params = getLearnableParameters(sacCritics(c), layerName);
                if ~isempty(params)
                    currentParams = getLearnableParameters(targetCritic, layerName);
                    if isequal(size(params{1}), size(currentParams{1}))
                        targetCritic = setLearnableParameters(targetCritic, layerName, params);
                    end
                end
            catch
                continue;
            end
        end
    end
    
    if c == 1, critic1 = targetCritic; else, critic2 = targetCritic; end
end

fprintf('Critic weight transfer complete.\n');