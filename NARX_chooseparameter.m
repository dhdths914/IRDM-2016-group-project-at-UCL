
clear
clc
%Load data set
load('data.mat');

%Load tranin data set 
input = tonndata(x(1:484,:),false,false);
target= tonndata(y(1:484),false,false);


%For different combination of parameter, calculation the average MSE of
%validation to choose the combination of parameter
for i=1:10
    for j=1:5
    for N=1:100

% Define the number of time delay 
Delays = 1:i;
Delays = 1:i;
%the number of hiddern layer
hiddenLayerSize = j;
%Construct the nautral network for NARX model
net = narxnet(Delays,Delays,hiddenLayerSize);

%Reconstruct the data into the time-dependent format
[inputs,inputStates,layerStates,targets] = preparets(net,input,{},target);

%According to the time, divide the data set into a small training data
%set(first 90% data)and validation data set(last 10% data)
net.divideFcn = 'divideblock';  
net.divideMode = 'value';  
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 0;%indicate that there is no test data

% Use Levenberg-Marquardt to train the network
net.trainFcn = 'trainlm'; 

%Calculate the Mean squred error 
net.performFcn = 'mse'; 


% Train the NARX nautral network
[net,tr] = train(net,inputs,targets,inputStates,layerStates);

%Use the net to predict the whole training data set 
outputs = net(inputs,inputStates,layerStates);


% Calculate Validation mse for every round
valTargets = gmultiply(targets,tr.valMask);
valPerformance(N)= perform(net,valTargets,outputs);
    end
   %Average the mse to get the final mse of every combination of parameters 
    valP(i,j)=mean(valPerformance);
    
    end
end

%Print the table of mse for every combination of parameters
valP