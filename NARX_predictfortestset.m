

clear
clc
%Load data set 
load('data.mat');
%Load the training data set 
input= tonndata(x(1:484,:),false,false);
target= tonndata(y(1:484),false,false);

%For the random initialize weight value and bias the result will be
%different,so average the prediction values over 100 time as the final
%prediction
for n=1:100
% Define the number of time delay 
Delays = 1:1;
Delays = 1:1;
%the number of hiddern layer
hiddenLayer =4;

%Construct the nautral network for NARX model
net = narxnet(Delays,Delays,hiddenLayer);



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

%Use this trained network to predict values in the test set 
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
testinput= tonndata(x(485:536,:),false,false);
testtarget = tonndata(y(485:536),false,false);
[xc,xic,aic,tc] = preparets(netc,testinput,{},testtarget);
yc = netc(xc,xic,aic);

ycc{n,:}=cell2mat(yc);
end

a=cell2mat(ycc);
b=mean(a);
 c=num2cell(b);
 %Calculate the mse of the prediction
closedLoopPerformance = perform(netc,tc,c)
%Calculate the correlation between prediction and original value
R=corr(cell2mat(tc)',b')

plotresponse(tc,c)

