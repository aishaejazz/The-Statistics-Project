%% Probabilistic & Statistical Modelling (II) Final Project - LDA
% Done by: Ayesha Ejaz
% Topic: Multi-model, Multivariate Analysis of Tactile Mental Imagery in Primary Somatosensory Cortex
%% Testing for Multivariate Normality

% (1) Defining directories or lists of beta file paths for each condition
condition1Dir = 'C:\Users\Medion\Downloads\Decoding_project\condition1';
condition2Dir = 'C:\Users\Medion\Downloads\Decoding_project\condition2';
condition3Dir = 'C:\Users\Medion\Downloads\Decoding_project\condition3';
roiMaskDir = 'C:\Users\Medion\Downloads\Decoding_project\ROI_file';

% (2) Listing all NIfTI files in the directories for each condition
condition1Files = dir(fullfile(condition1Dir, '*.nii'));
condition2Files = dir(fullfile(condition2Dir, '*.nii'));
condition3Files = dir(fullfile(condition3Dir, '*.nii'));
numROIs = 5;

% (3) Initializing cell arrays to store beta values for each condition and ROI
allBetaValuesCondition1 = cell(1, numROIs);
allBetaValuesCondition2 = cell(1, numROIs);
allBetaValuesCondition3 = cell(1, numROIs);

% (4) Looping through NIfTI files for Condition 1
for fileIdx = 1:numel(condition1Files)
    % (5) Getting the filename
    filename = condition1Files(fileIdx).name;
    
    % (6) Checking if the filename indicates it's a beta map
    if startsWith(filename, 'beta')
        niftiFile = fullfile(condition1Dir, filename);
        nifti = spm_vol(niftiFile);
        niftiData = spm_read_vols(nifti);
        
        % (7) Looping through each ROI
        for roiIdx = 1: numROIs
            roiMaskFile = fullfile(roiMaskDir, ['ROI', num2str(roiIdx), '.nii']);
            roiMask = spm_vol(roiMaskFile);
            roiMaskData = spm_read_vols(roiMask);
            
            % (8) Extracting beta values from the ROI
            betaValues = niftiData(roiMaskData > 0);
            
            % (9) Storing the beta values in the cell array for Condition 1
            allBetaValuesCondition1{1, roiIdx} = [allBetaValuesCondition1{1, roiIdx}; betaValues];
        end
    end
end

% Repeating the same process for Condition 2 and Condition 3
for fileIdx = 1:numel(condition2Files)
    % (1) Getting the filename
    filename = condition2Files(fileIdx).name;
    
    % (2) Checking if the filename indicates it's a beta map
    if startsWith(filename, 'beta')
        niftiFile = fullfile(condition2Dir, filename);
        nifti = spm_vol(niftiFile);
        niftiData = spm_read_vols(nifti);
        
        % (3) Looping through each ROI
        for roiIdx = 1: numROIs
            roiMaskFile = fullfile(roiMaskDir, ['ROI', num2str(roiIdx), '.nii']);
            roiMask = spm_vol(roiMaskFile);
            roiMaskData = spm_read_vols(roiMask);
            
            % (4) Extracting beta values from the ROI
            betaValues = niftiData(roiMaskData > 0);
            
            % (5) Storing the beta values in the cell array for Condition 1
            allBetaValuesCondition2{1, roiIdx} = [allBetaValuesCondition2{1, roiIdx}; betaValues];
        end
    end
end

% For condition 3
for fileIdx = 1:numel(condition3Files)
    % (1) Getting the filename
    filename = condition3Files(fileIdx).name;
    
    % (2) Checking if the filename indicates it's a beta map
    if startsWith(filename, 'beta')
        niftiFile = fullfile(condition3Dir, filename);
        nifti = spm_vol(niftiFile);
        niftiData = spm_read_vols(nifti);
        
        % (3) Looping through each ROI
        for roiIdx = 1: numROIs
            % (4) Loading the ROI mask
            roiMaskFile = fullfile(roiMaskDir, ['ROI', num2str(roiIdx), '.nii']);
            roiMask = spm_vol(roiMaskFile);
            roiMaskData = spm_read_vols(roiMask);
            
            % (5) Extracting beta values from the ROI
            betaValues = niftiData(roiMaskData > 0);
            
            % (6) Storing the beta values in the cell array for Condition 1
            allBetaValuesCondition3{1, roiIdx} = [allBetaValuesCondition3{1, roiIdx}; betaValues];
        end
    end
end

% Vertically concatenating all the ROIs with each column corresponding to
% one of the conditions
allBetaValues = [allBetaValuesCondition1{1,1} allBetaValuesCondition2{1,1} allBetaValuesCondition3{1,1}; allBetaValuesCondition1{1,2} allBetaValuesCondition2{1,2} allBetaValuesCondition3{1,2}; allBetaValuesCondition1{1,3} allBetaValuesCondition2{1,3} allBetaValuesCondition3{1,3};allBetaValuesCondition1{1,4} allBetaValuesCondition2{1,4} allBetaValuesCondition3{1,4}; allBetaValuesCondition1{1,5} allBetaValuesCondition2{1,5} allBetaValuesCondition3{1,5}];
% Performing Henze Zirkler test
HZmvntest(allBetaValues)

%% Testing for covariance matrices 
Z = [ones(630,1) allBetaValuesCondition1{1,1} allBetaValuesCondition2{1,1} allBetaValuesCondition3{1,1}; ones(942,1)*2 allBetaValuesCondition1{1,2} allBetaValuesCondition2{1,2} allBetaValuesCondition3{1,2}]
% Testing for two ROIs, namely BA1 and BA2 for the three imagery conditions
MBoxtest(Z,0.05);

%% The Decoding Toolbox (TDT) was used for performing LDA. 

%% First, setting the defaults and defining the analysis 

clearvars;
clc;
data_dir = 'C:\Users\Medion\Downloads\Decoding_project';
addpath(data_dir); % Adding path to all subject folders
addpath('C:\Users\Medion\Downloads\tdt_3.999E2\decoding_toolbox'); % Adding path to tdt toolbox
assert(~isempty(which('decoding_defaults.m', 'function')), 'TDT not found in path, please add');
addpath('C:\spm12\spm12'); % Adding path to SPM 12
assert((~isempty(which('spm.m', 'function')) || ~isempty(which('BrikInfo.m', 'function'))) , 'Neither SPM nor AFNI found in path, please add (or remove this assert if you really dont need to read brain images)');

clear cfg;

for subj = 1:10 % Iterating over all subjects

decoding_defaults; % Setting the defaults

cfg.analysis = 'ROI'; % Defining the analysis method
cfg.searchlight.radius = 3; % The unit by default is voxels
  
cfg.results.dir = sprintf('Decoding_project/results_ImagPress_vs_ImagVibro/sub-%03d', subj); % Defining where results would be saved
cfg.results.overwrite = 1; % In case you're running the analysis again and don't want the previous results, otherwise set to 0.

%% Second, getting the file names, labels and run number of each brain image
%% file to use for decoding.


subjfolder = sprintf('/sub-%03d/1st_level_good_bad_Imag/', subj); % Indicating path to subject folders

beta_loc = fullfile(data_dir, subjfolder); % Defining the path to SPM.mat and all related beta files

labelname1 = 'ImagPress'; % Specifying the label names that were given to the regressors of interest. This was rewritten two more times to perform pairwise classification between ImagPress, ImagFlutt and ImagVibro.
labelname2 = 'ImagVibro';


cfg.files.mask = {'C:\Users\Medion\Downloads\Decoding_project\rois\rPSC_1_TR50_right_CUT_Stim_vs_Null.nii', 'C:\Users\Medion\Downloads\Decoding_project\rois\rPSC_2_TR50_right_CUT_Stim_vs_Null.nii', 'C:\Users\Medion\Downloads\Decoding_project\rois\rPSC_3b_TR50_right_CUT_Stim_vs_Null.nii', 'C:\Users\Medion\Downloads\Decoding_project\rois\rSII_TR50_left_CUT_Stim_vs_Null.nii', 'C:\Users\Medion\Downloads\Decoding_project\rois\rSII_TR50_right_CUT_Stim_vs_Null.nii'};
% Multiple ROI masks from regions BA1, BA2, BA3b, ipsilateral S2 and contralateral S2 to define which voxels to use in the analysis

regressor_names = design_from_spm(beta_loc); % Extracting all beta names and corresponding run numbers from the SPM.mat file


cfg = decoding_describe_data(cfg,{labelname1 labelname2},[1 -1],regressor_names,beta_loc); % Extracting the file names and run numbers of each label


%% Third, creating the design for decoding analysis


 cfg.design = make_design_cv(cfg); % Creating a leave-one-run-out cross-validation design


%% Fourth, setting the additional parameters manually


cfg.verbose = 1; % How much output you want to see on the screen while the program is running. In this case, 0 = no output.

cfg.decoding.method = 'classification'; % This is our default anyway.

cfg.decoding.train.classification.model_parameters.shrinkage = 'lw2'; % Sets the regularization method to 'lw2,' which likely refers to L2 (ridge) regularization. Regularization methods like L2 regularization add a penalty term to the loss function during training to prevent overfitting by discouraging overly complex models. In the context of linear classification or regression, L2 regularization adds a term that penalizes large coefficients, encouraging the model to use smaller weights for features. This helps improve the model's generalization performance on unseen data.

cfg.results.output = {'accuracy_minus_chance'}; % Chance value is 50

cfg.decoding.software = 'lda'; 

%% Fifth, plotting


cfg.plot_selected_voxels = 0; % In this case, plot nothing online

cfg.plot_design = 1; % This will call display_design(cfg);

display_design(cfg); % Allows you to look at your design after plotting

%% Sixth, Running the decoding analysis

results = decoding(cfg);
end 

%% Seventh, performing group analysis
%% Part-1: Extracting Data

% Getting all the pairwise classification accuracies in one matrix for all
% subjects with each row being one ROI and each column a separate subject.

subjects = [1 2 3 4 5 6 7 8 9 10];
Group_Results_ImagPress_vs_ImagFlutt = [];
Group_Results_ImagPress_vs_ImagVibro = [];
Group_Results_ImagFlutt_vs_ImagVibro = [];



for subject=subjects

    subject = num2str(subject);

    ROI_Results1 = load([pwd '\results_ ImagPress_vs_ImagFlutt\sub-00' subject '/res_accuracy_minus_chance.mat']);
    ROI_Results2 = load([pwd '\results_ImagPress_vs_ImagVibro\sub-00' subject '/res_accuracy_minus_chance.mat']);
    ROI_Results3 = load([pwd '\results_ImagFlutt_vs_ImagVibro\sub-00' subject '/res_accuracy_minus_chance.mat']);

    Group_Results_ImagPress_vs_ImagFlutt = [Group_Results_ImagPress_vs_ImagFlutt, ROI_Results1.results.accuracy_minus_chance.output];
    Group_Results_ImagPress_vs_ImagVibro = [Group_Results_ImagPress_vs_ImagVibro, ROI_Results2.results.accuracy_minus_chance.output];
    Group_Results_ImagFlutt_vs_ImagVibro = [Group_Results_ImagFlutt_vs_ImagVibro, ROI_Results3.results.accuracy_minus_chance.output];

end
%% Part-2: Statistics for each pairwise classification alone

% Calculating the mean across all subjects for the five ROIs
ImagPress_vs_ImagFlutt_mean = mean(Group_Results_ImagPress_vs_ImagFlutt, 2);
ImagPress_vs_ImagVibro_mean = mean(Group_Results_ImagPress_vs_ImagVibro, 2);
ImagFlutt_vs_ImagVibro_mean = mean(Group_Results_ImagFlutt_vs_ImagVibro, 2);

% Getting the index of the ROI for each subject in which the decoding
% accuracy was highest
[~, ImagPress_vs_ImagFlutt_highest]= max(Group_Results_ImagPress_vs_ImagFlutt, [], 1);
[~,ImagPress_vs_ImagVibro_highest] = max(Group_Results_ImagPress_vs_ImagVibro, [], 1);
[~,ImagFlutt_vs_ImagVibro_highest] = max(Group_Results_ImagFlutt_vs_ImagVibro, [], 1);

% Getting the index of the ROI in which the accuracy was high for the most
% subjects
ImagPress_vs_ImagFlutt_highest_mode= mode(ImagPress_vs_ImagFlutt_highest);
ImagPress_vs_ImagVibro_highest_mode = mode(ImagPress_vs_ImagVibro_highest);
ImagFlutt_vs_ImagVibro_highest_mode = mode(ImagFlutt_vs_ImagVibro_highest);

% Performing ttest for each ROI and each classifier:
Group1 = Group_Results_ImagPress_vs_ImagFlutt;
Group2 = Group_Results_ImagPress_vs_ImagVibro; 
Group3 = Group_Results_ImagFlutt_vs_ImagVibro; 

% (1) Initializing a cell array to store results structures for each group
allGroupsROI = cell(3, 1);

% (2) Performing one-sample t-test for each ROI and each group
for groupIndex = 1:3
    Group = eval(['Group', num2str(groupIndex)]); % Getting the current data matrix
    
    % (3) Initializing a results structure for the current matrix
    Group_results = struct();
    
    % (4) Performing one-sample t-test for each ROI (each row of the matrix)
    for roiIndex = 1:size(Group, 1)
        roiData = Group(roiIndex, :);
        
        % (5) Performing one-sample t-test comparing ROI data to zero (chance)
        [h, p, ci, stats] = ttest(roiData);
        
        % (6) Storing results in the structure
        Group_results(roiIndex).roiData = roiData;
        Group_results(roiIndex).t_statistic = stats.tstat;
        Group_results(roiIndex).df = numel(roiData) - 1; 
        Group_results(roiIndex).p_value = p;
        Group_results(roiIndex).ci = ci;
        Group_results(roiIndex).h = h; % True if above chance, False if not
    end
    
    % (7) Storing the results structure in the cell array
    allGroupsROI{groupIndex} = Group_results;
end

% Drawing the figure for allGroupsROI
% (1) Defining category colors
categoryColors = {"#A2142F", "#77AC30", "#0072BD"};  % Adjust colors as needed

% (2) Creating a figure with three subplots
figure;

for groupIndex = 1:3
    % (3) Getting the results for the current matrix
    Group_results = allGroupsROI{groupIndex};
    
    % (4) Creating a subplot for the current matrix
    subplot(1, 3, groupIndex); % 1 row, 3 columns, current subplot
    
    % (5) Ploting the pvalues with category-specific colors
    p_value = [Group_results.p_value];
    bar(p_value, 'FaceColor', categoryColors{groupIndex});
    xlabel('ROI Index');
    ylabel('P-value');
    title(['Group ' num2str(groupIndex)]);

     if groupIndex == 3
        hold on;
        second_value = p_value(2); 
        plot(2, second_value + 0.02, 'k*', 'MarkerSize', 10); 
        hold off;
     end 
end 
sgtitle('Comparison of P-values across all ROIs for each classifier');

%% Part-3: Comparing the three different classifications

% Performing ttest between the pairwise classifiers
[h, p, ci, stats] = ttest2(ImagPress_vs_ImagVibro_mean, ImagPress_vs_ImagFlutt_mean); % Pairwise classifiers ttest-1
disp("T-test results:");
disp("Hypothesis test result (h): " + h);
disp("p-value (p): " + p);
disp("Confidence interval (ci): " + ci);
disp("T-test statistics (stats):");
disp(stats);


[h, p, ci, stats] = ttest2(ImagPress_vs_ImagVibro_mean, ImagFlutt_vs_ImagVibro_mean); %Pairwise classifiers ttest-2
disp("T-test results:");
disp("Hypothesis test result (h): " + h);
disp("p-value (p): " + p);
disp("Confidence interval (ci): " + ci);
disp("T-test statistics (stats):");
disp(stats);

[h, p, ci, stats] = ttest2(ImagPress_vs_ImagFlutt_mean, ImagFlutt_vs_ImagVibro_mean); % Pairwsie classifiers ttest-3
disp("T-test results:");
disp("Hypothesis test result (h): " + h);
disp("p-value (p): " + p);
disp("Confidence interval (ci): " + ci);
disp("T-test statistics (stats):");
disp(stats);

% ttest for each roi across all subjects for a group of two classifiers
% (1) Defining the data matrices for the comparisons
dataMatrix1 = Group_Results_ImagPress_vs_ImagFlutt;
dataMatrix2 = Group_Results_ImagPress_vs_ImagVibro;
dataMatrix3 = Group_Results_ImagFlutt_vs_ImagVibro;

% (2) Creating a cell array to hold data matrices for the comparisons
dataMatrices = {dataMatrix1, dataMatrix2, dataMatrix3};

% (3) Initializing a results structure
results2 = struct();

% (4) Performing t-test for each row separately and each pairwise comparison
numComparisons = length(dataMatrices);
numRows = size(dataMatrix1, 1); % Assuming all data matrices have the same number of rows

for comparisonIndex1 = 1:numComparisons
    dataMatrix1 = dataMatrices{comparisonIndex1};
    
    for comparisonIndex2 = comparisonIndex1+1:numComparisons
        dataMatrix2 = dataMatrices{comparisonIndex2};
        
        for rowIndex = 1:numRows
            value1 = dataMatrix1(rowIndex, :);
            value2 = dataMatrix2(rowIndex,:);
            
            % (5) Performing t-test
            [h, p, ci, stats] = ttest2(value1, value2);
            t_statistic = stats.tstat;
            
            % (6) Calculating Cohen's d
            n1 = sum(~isnan(value1)); % Sample size for group 1
            n2 = sum(~isnan(value2)); % Sample size for group 2
            d = t_statistic / sqrt((n1 + n2) / 2); % Assuming equal sample sizes
            
            % (7) Storing results in the structure
            results2(end+1).comparisonIndex1 = comparisonIndex1;
            results2(end).comparisonIndex2 = comparisonIndex2;
            results2(end).rowIndex = rowIndex;
            results2(end).t_statistic = t_statistic;
            results2(end).cohen_d = d;
            results2(end).ci = ci; 
            results2(end).p_value = p; 
            results2(end).h = h


        end
    end
end

% (8) The first row is empty for some reason, so just removing that
results2(1) = [];

% Bonferroni Correction
% (1) Getting the p-values
pValues = [results2.p]
numComparisons = numel(pValues);

% (2) Desired family-wise error rate (usually 0.05)
desiredFWER = 0.05;

% (3) Calculating adjusted alpha (Bonferroni-adjusted significance level)
adjustedAlpha = desiredFWER / numComparisons;

% (4) Determining which tests are significant after Bonferroni correction
significantTests = pValues < adjustedAlpha;

% Creating a figure from the results(2) struct
% (1) Extracting cohen_d values from the results2 struct
cohen_d = [results2.cohen_d];

% (2) Creating an error barplot

categories = {'1 v 2', '1 v 2', '1 v 2', '1 v 2', '1 v 2', ...
              '1 v 3', '1 v 3', '1 v 3', '1 v 3', '1 v 3', ...
              '2 v 3', '2 v 3', '2 v 3', '2 v 3', '2 v 3'};

% (3) Definining unique categories
uniqueCategories = unique(categories);
numCategories = numel(uniqueCategories);

% (4) Definining colors for each category
categoryColors = {"#A2142F", "#77AC30", "#0072BD"};  % Adjust colors as needed

% (5) Creating a bar graph with different colors for each category
figure;
hold on;

% (6) Looping through unique categories and setting the color for each category
for i = 1:numel(uniqueCategories)
    categoryData = cohen_d(strcmp(categories, uniqueCategories{i}));
    bar(find(strcmp(categories, uniqueCategories{i})), categoryData, 'FaceColor', categoryColors{i});
end

hold off;
% (7) Setting x-axis tick labels as well as x and y labels and the title
xticks(1:numel(cohen_d));
xticklabels(categories);
ylabel('Effect size');
xlabel('Categories of Pairwise Classification');
title('Comparison of performance of Pairwise Classifiers');

% (8) Rotating x-axis labels for better readability
xtickangle(45);

% (9) Extracting the confidence intervals from the results structure and
% putting it in a suitable format for including an error bar
ci_results = [results2.ci];
ciLower = ci_results(:, [1 3 5 7 9 11 13 15 17 19 21 23 25 27 29]);
ciUpper = ci_results(:, [2 4 6 8 10 12 14 16 18 20 22 24 26 28 30]);
ciUpper = (ciUpper)';
ciLower = (ciLower)';
error = (ciUpper - ciLower)/3.92; % 3.92 for 95% CI
hold on;
errorbar(1:numel(cohen_d), cohen_d, error, 'k', 'LineStyle', 'none', 'CapSize', 0);
seventhValueIndex = 7; 
plot(seventhValueIndex, cohen_d(seventhValueIndex), 'k*', 'MarkerSize', 10);
legend('Cohen_d 1 v 2', 'Cohen_d 1 v 3', 'Cohen_d 2 v 3', 'error');
hold off;

%% The end. 
