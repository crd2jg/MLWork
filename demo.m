%% Classification Learner Demo
%This demo requires both importiris.m and trainClassifier.m
%This demo uses the iris data set included with MATLAB
%A model is either trained using the classificationLearner App or via
%generated code

clear %clear the workspace
%% Import Iris data

irisdata = importiris(); %import the data

vizOn = true; 
%% Visualization
% Create a 4x4 grid of plots to show the difficulty in classifying the 
% species of iris
if vizOn 
    figure('units','normalized','outerposition',[0 0 1 1])
    hold on
    for n=1:16
        subplot(4,4,n)
        
        nx = mod(n,4);
        if nx == 0
            nx = 4;
        end
        ny = ceil(n/4);
        
        x=table2array(irisdata(:,nx));
        y=table2array(irisdata(:,ny));
        gscatter(x,y,irisdata.Species)
        labs = {"Sepal Length","Sepal Width","Petal Length","Petal Width"};
        
        if nx == 1
            ylabel(labs(ny))
        else
            ylabel("")
        end
        
        if ny == 4
            xlabel(labs(nx))
        else
            xlabel("")
        end
    end
    hold off
end
clear x y species labs meas n nx ny
%% Take subset for testing model

subsetSize = 20; %take 20 iris for testing
rng(42)
[subset,idx] = datasample(irisdata,subsetSize,'Replace',false);

trainingset = irisdata(~ismember(1:150,idx),:);
clear idx
clear irisdata

%% ClassificationLearner App

%classificationLearner 
%Use the UI to train multiple models and select the best one
%export it or generate code to train it
%% Use Exported Code
[trainedClassifier, validationAccuracy] = trainClassifier(trainingset);
%alternatively train a classifier using generated code
%% Classify remaining data
%Use the classifier/model to predict the species for our 20 iris
predictions = trainedClassifier.predictFcn(subset(:,1:4));

%% Check predictions
correctClass = table2cell(subset(:,5));

accurate = strcmp(predictions,correctClass);

percentCorrect = sum(accurate)/length(accurate)*100;
disp(['Accuracy of the trained Model for the subset is:  ',...
    num2str(percentCorrect),'%.'])
disp('All predicted values');
fprintf('\n     Predicted       Correct\n');
disp([predictions, correctClass]);

disp('Incorrect Values');
fprintf('     Predicted       Correct\n');
diff = [predictions(~accurate), correctClass(~accurate)];
disp(diff)
%% Confusion Plot
% create arrays for format of plotconfusion function
predictArray = [ismember(predictions, 'setosa') ,...
    ismember(predictions, 'versicolor'),...
    ismember(predictions, 'virginica')];

correctArray =[ismember(correctClass, 'setosa') ,...
    ismember(correctClass, 'versicolor'),...
    ismember(correctClass, 'virginica')];

figure
plotconfusion(double(correctArray'),double(predictArray'));

%% Neural Network
net = patternnet(18); %number of nodes isn't important for us right now

%prepare data into format for net
trainingsetNet = table2array(trainingset(:,1:4));
idwlabs = table2cell(trainingset(:,5));

idNet = [ismember(idwlabs, 'setosa') ,...
    ismember(idwlabs, 'versicolor'),...
    ismember(idwlabs, 'virginica')];
%train network
net = train(net,trainingsetNet',idNet');

%% Predict using Neural Network
%predict using network
predictionsNN = net(table2array(subset(:,1:4))');
%% Display Confusion Matrix

figure
plotconfusion(double(correctArray'),predictionsNN)