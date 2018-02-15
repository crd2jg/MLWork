function irisdata = importiris()
%% Loads the Iris data set and returns a table of the data
load fisheriris.mat meas species

%% Convert to tables
meas = array2table(meas);
species = cell2table(species);

%% Combine tables and rename Variables
irisdata = [meas,species];
irisdata.Properties.VariableNames = {'SepalLength','SepalWidth','PetalLength','PetalWidth','Species'};
end