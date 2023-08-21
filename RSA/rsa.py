## Probabilistic & Statistical Modelling (II) Final Project - RSA (Python)
# Done by: Lucy Lucy Roellecke
# Topic: Multi-model, Multivariate Analysis of Tactile Mental Imagery in Primary Somatosensory Cortex
# fMRI data is taken from a study by Nierhaus et al. (2023)


## Setup 

# Import all functions from functions script (should be in the same directory as this script)
from rsa_functions import *
import rsatoolbox
import rsatoolbox.rdm as rsr
import scipy.stats as stats


## Variables

# Script should be in directory /code/ and data in another directory /data/
datapath = "/Volumes/INTENSO/data/"
# Path where to save the results of the analysis
resultpath = "../analysis/"

# Subjects (N = 10)
subjects = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

# Conditions
all_conditions = ["stim_press", "stim_flutt", "stim_vibro", 
                  "imag_press", "imag_flutt", "imag_vibro"]
selected_conditions = ["stim_press", "stim_flutt", "stim_vibro",
                       "imag_press", "imag_flutt", "imag_vibro"]

# Runs (6)
runs = ["01", "02", "03", "04", "05", "06"]

# ROIs
# five regions of interest as defined by the original paper 
# (intersection of stimulation vs. baseline contrast and anatomic masks)
#       rPSC_1      :   contralateral (right) primary somatosensory cortex BA 1
#       rPSC_2      :   contralateral (right) primary somatosensory cortex BA 2
#       rPSC_3b     :   contralateral (right) primary somatosensory cortex BA 3b
#       rSII_right  :   contralateral (right) secondary somatosensory cortex
#       rSII_left   :   ipsilateral (left) secondary somatosensory cortex
regions_of_interest = ["rPSC_2", "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]


## Data formatting

# Initiate 6D array to fill with beta values of all subjects
formatted_data = np.empty(
    (79, 95, 79, len(selected_conditions), len(runs), len(subjects)))
for index, subject in enumerate(subjects):
    folder_path = os.path.join(
        datapath, f"sub-{subject}", "1st_level_good_bad_Imag")
    formatted_data[:, :, :, :, :, index] = format_data_for_subject(
        folder_path, runs, selected_conditions, all_conditions)

# Formatted_data is now a 6D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: selected runs (1-6)
#       6th dimension: 10 participants

# Average over runs
averaged_data = average_over_runs(formatted_data)


## RSA 

# Representational similarity analysis for each region of interest
# Initiate empty lists to fill with RDMs for each region and condition
stimulation_RDMs_euclidean_all_regions = []
imagery_RDMs_euclidean_all_regions = []
all_RDMs_euclidean_all_regions = []

# Loop over region of interests and compute a RSA for each region separately
for region in regions_of_interest:
    # apply roi mask to data so only voxels of that roi are analyzed
    voxels_from_region = get_voxels_from_region_of_interest(
        region, datapath)
    # index those voxels in our main data array and rearrange dimensions of array
    # to fit dataset object
    data_from_region = rearrange_array(voxels_from_region, averaged_data)

    # Data_from_region is now a 3D array with the following dimensions
    #       1st dimension: 6 conditions/stimulus types
    #       2nd dimension: roi voxels
    #       3rd dimension: 10 participants

    # Transform data into dataset object for using the RSAToolbox by Schütt et al., 2019
    conditions_key = 'conditions'
    region_datasets = create_rsa_datasets(data_from_region, len(subjects), conditions_key)

    # Select data only from conditions 1:3 (stimulation) and 4:6 (imagery)
    stimulation_conditions = [conditions_key + str(number)
                              for number in range(1, 4)]
    imagery_conditions = [conditions_key + str(number)
                          for number in range(4, 7)]
    
    stimulation_data = []
    imagery_data = []
    for dataset in region_datasets:
        stimulation_sub_dataset = dataset.subset_obs(by=conditions_key, 
                                                     value=stimulation_conditions)
        imagery_sub_dataset = dataset.subset_obs(by=conditions_key, 
                                                 value=imagery_conditions)
        stimulation_data += [stimulation_sub_dataset]
        imagery_data += [imagery_sub_dataset]


    ## Calculate RDMS

    # Calculates a representational dissimilarity matrix for stimulation data, 
    # For imagery data and for all data
    # Euclidean distance
    stimulation_RDM_euclidean = rsr.calc_rdm(
        stimulation_data, method='euclidean', descriptor=conditions_key)
    imagery_RDM_euclidean = rsr.calc_rdm(
        imagery_data, method='euclidean', descriptor=conditions_key)
    all_RDM_euclidean = rsr.calc_rdm(
        region_datasets, method='euclidean', descriptor=conditions_key)
    
    # Add RDMs to list for all regions
    stimulation_RDMs_euclidean_all_regions += [stimulation_RDM_euclidean]
    imagery_RDMs_euclidean_all_regions += [imagery_RDM_euclidean]
    all_RDMs_euclidean_all_regions += [all_RDM_euclidean]

    # Print RDMs and plot them for manual inspection
    show_debug_for_rdm(stimulation_RDM_euclidean)
    show_debug_for_rdm(imagery_RDM_euclidean)
    show_debug_for_rdm(all_RDM_euclidean)
    input("Press Enter to continue...")


    ## Compare RDMS  

    # Compares both RDMs (stimulation and imagery) and calculates their similiarity
    # pearson correlation
    method = 'corr'
    similiarities = []
    for subject in subjects:
        similiarity = rsatoolbox.rdm.compare(
            stimulation_RDM_euclidean.subset('subjects', int(subject)),
            imagery_RDM_euclidean.subset('subjects', int(subject)),
            method = method)
        similiarities += [(similiarity[0][0])]
    
    # Other possible similiarity measures:
    #               Pearson ('corr')
    #               Cosine ('cosine')
    #               whitened comparison methods ('corr_cov' or 'cosine_cov')
    #               Kendall's tau ('tau-a')
    #               Spearman's rho ('rho-a')

    # Test similiarity for significance with a one-sample t-test
    # when using Pearson's correlation to compare RDMs, the null hypothesis
    # is a correlation of 0, meaning that two RDMs are not similiar
    # so we use a population mean of 0 as our null hypothesis
    # when using a different similiarity measure, 
    # this needs to be adjusted at popmean=xx
    significance = stats.ttest_1samp(similiarities, popmean=0, alternative='greater')
    significance_report = (method + ' = ' + str(round(np.mean(similiarities), 3)) + 
                           ' (T = ' + str(round(significance.statistic, 3)) + ', p = ' 
                           + str(round(significance.pvalue, 3)) + ', df = ' 
                           + str(significance.df) + ')')

    print('The average similarity of stimulation and imagery RDMs across ' +
          str(len(subjects)) + ' subjects in ' + str(region) + ' is: ' 
          + str(significance_report))


    ## Save results

    # Save representational dissimiliarity matrices and corresponding figures
    for subject in subjects:
        save_rdm_results(resultpath, region, 'stimulation', 'euclidean',
                        stimulation_RDM_euclidean.get_matrices()[int(subject)-1,:,:],
                        subject)
        save_rdm_results(resultpath, region, 'imagery', 'euclidean',
                        imagery_RDM_euclidean.get_matrices()[int(subject)-1,:,:],
                        subject)
        save_rdm_results(resultpath, region, 'all', 'euclidean',
                        all_RDM_euclidean.get_matrices()[int(subject)-1,:,:],
                        subject)

    # Save representational similiarity analysis results
    save_rsa_results(resultpath, region, 'stim_imag', method, similiarities)
    save_rsa_results(resultpath, region, 'stim_imag', 'ttest', significance_report)


## Compare RDMS across regions

# Compare similiarity of imagery and perception in the different regions of interest

# Read in rsa results of different regions as numpy array
# Initiate empty array to fill with similiarity values for regions
all_similiarity_values = np.empty((len(regions_of_interest),len(subjects)))
for index,region in enumerate(regions_of_interest):
    rsa_path = os.path.join(resultpath, region, 'rsa', 'stim_imag_rsa_corr.txt')
    region_similiarity_values = np.genfromtxt(rsa_path, delimiter = ',')
    all_similiarity_values[index,:] = region_similiarity_values[0:-1]

# Perform a one-way anova to test whether the rois have the same population mean
# Null hypothesis: group means are equal
anova_results = stats.f_oneway(all_similiarity_values[0],
                               all_similiarity_values[1],
                                all_similiarity_values[2],
                                 all_similiarity_values[3],
                                  all_similiarity_values[4])

anova_report = ('F(' + str(len(regions_of_interest)-1) + ',' + 
                str(len(subjects)-len(regions_of_interest)) 
                + ') = ' + str(round(anova_results.statistic,2)) + 
                ', p = ' + str(round(anova_results.pvalue, 3)))
print('Repeated-measures analysis of variance (ANOVA) did not show' + 
      'any significant differences in similiarity measures between ' +
      'regions of interest ' + anova_report + '.')


## Save results

# Make a new directory for the region and anova
rsa_path = os.path.join(resultpath, "anova")
if os.path.exists(rsa_path) == False:
    os.makedirs(rsa_path)

# Save anova results as text file
filename = os.path.join(rsa_path, "similiarities_anova.txt")
if os.path.exists(filename) == True:
    os.remove(filename)
file = open(filename, 'w')
file.write(anova_report)
file.close()
