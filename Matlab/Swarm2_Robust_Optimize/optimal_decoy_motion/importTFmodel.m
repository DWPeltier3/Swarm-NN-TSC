clc; clear all;

%% Import TF NN into Matlab

model = "CNmc20_10v1C10_perL"; % CNmc20_10v1C10_perL, CNmc20_10v1C10_combDM, CNmc20_10v1C10_semi, CNNEXmc20comb10v10, CNNEXmc20comb10v10_r40k, CNNEXmcFULLcomb10v10, CNmcFull, LSTMMmc20, LSTMMmcFull
temp_modelFolder = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/models/temp/" + model;
net = importNetworkFromTensorFlow(temp_modelFolder);

% %% REPLACE FAULTY MASKING LAYER
% % Define the custom masking layer
% maskValue = 0.0;
% customMaskingLayer = CustomMaskingLayer(maskValue, 'custom_masking');

% % Define the new sequence input layer
% inputFeatureSize = 40; % size for the feature dimension
% inputTimeSize = 55; % size for the time dimension
% % newInputLayer = sequenceInputLayer(inputSize, 'Name', 'sequence_input'); % inputsize = num_features
% newInputLayer = sequenceInputLayer(inputFeatureSize, MinLength=inputTimeSize, Name='sequence_input');
% 
% % Replace the Masking layer with the custom masking layer
% layers = layerGraph(net);
% layers = replaceLayer(layers, 'masking', customMaskingLayer);
% layers = replaceLayer(layers, 'input_1', newInputLayer); % Replace 'input_1' with the name of your current input layer
% 
% % Convert to dlnetwork
% net = dlnetwork(layers);

% Save the network
modelFolder = "/Users/DWPeltier3/Library/CloudStorage/OneDrive-NavalPostgraduateSchool/1Documents/0Research_Code/swarm/models/" + model;
save(modelFolder + ".mat", 'net');  % Assume 'net' is your imported model
analyzeNetwork(net);
