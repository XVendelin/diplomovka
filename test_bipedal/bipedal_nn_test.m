%% Load model data and model
clear all; clc;
robotParametersRL;

mdl = "rlWalkingBipedRobot";
load_system(mdl);
set_param(mdl, 'SimulationMode', 'accelerator')
% open_system(mdl)

%% Create Environment Object

% Create the observation specification.

numObs = 29;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "observations";

% Create the action specification.

numAct = 6;
actInfo = rlNumericSpec([numAct 1],LowerLimit=-1,UpperLimit=1);
actInfo.Name = "foot_torque";

% Create the environment object for the walking robot model.

blk = mdl + "/RL Agent";
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
env.ResetFcn = @(in) walkerResetFcn(in, ...
    upper_leg_length/100, ...
    lower_leg_length/100, ...
    h/100);

%% Select and Create Agent for Training

% rng(0,"twister");

AgentSelection = "DDPG";
switch AgentSelection
    case "DDPG"
        agent = createDDPGAgent(numObs,obsInfo,numAct,actInfo,Ts);
    case "TD3"
        agent = createTD3Agent(numObs,obsInfo,numAct,actInfo,Ts);
    otherwise
        disp("Assign AgentSelection to DDPG or TD3")
end

% Get the actor and critic neural networks from the agent

actor = getActor(agent);
critic = getCritic(agent);
actorNet = getModel(actor);
criticNet = getModel(critic(1));

% summary(actorNet);
% plot(actorNet);

% summary(criticNet);
% plot(criticNet);

%% Specify Training Options and Train Agent

maxEpisodes = 4000;
maxSteps = floor(Tf/Ts);
trainOpts = rlTrainingOptions(...
    MaxEpisodes=maxEpisodes,...
    MaxStepsPerEpisode=maxSteps,...
    ScoreAveragingWindowLength=250,...
    StopTrainingCriteria="none",...
    SimulationStorageType= "file",...
    SaveSimulationDirectory=AgentSelection+"Sims");

trainOpts.UseParallel = true;
trainOpts.ParallelizationOptions.Mode = "async";

% rng(0,"twister");

%% Training

doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load a pretrained agent for the selected agent type.
    if strcmp(AgentSelection,"DDPG")
       load(fullfile("DDPGAgent","run1.mat"),"agent")
    else
       load(fullfile("TD3Agent","run1.mat"),"agent")
    end  
end

%% Simulate Trained Agents

% rng(0,"twister");

simOptions = rlSimulationOptions(MaxSteps=maxSteps);
experience = sim(env,agent,simOptions);


%% Compare Agent Performance

% rng(0,"twister");

comparePerformance("DDPGAgent","TD3Agent")