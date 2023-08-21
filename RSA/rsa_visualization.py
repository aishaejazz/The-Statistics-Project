# Python script visualizing the results of the rsa performed by rsa.py

## Setup
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rsa_functions import *

# Path where to the results of the analysis are (and where to save the plots as well)
resultpath = "../analysis/"

# Visualization steps
steps = ["plot_rdms_all_conds"]
# "plot_rdms", "plot_rdm_comparison", "plot_rsm_cross_validated"

# Subjects (N = 10)
subjects = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"]

# Runs (6)
runs = ["01", "02", "03", "04", "05", "06"]

# Conditions
all_conditions = ["stim_press", "stim_flutt", "stim_vibro", 
                  "imag_press", "imag_flutt", "imag_vibro"]
stimulation_conditions = ["stim_press", "stim_flutt", "stim_vibro"]
imagery_conditions = ["imag_press", "imag_flutt", "imag_vibro"]

# ROIs
# five regions of interest as defined by the original paper (intersection of stimulation vs. baseline contrast and anatomic masks)
#       rPSC_1      :   contralateral (right) primary somatosensory cortex BA 1
#       rPSC_2      :   contralateral (right) primary somatosensory cortex BA 2
#       rPSC_3b     :   contralateral (right) primary somatosensory cortex BA 3b
#       rSII_right  :   contralateral (right) secondary somatosensory cortex
#       rSII_left   :   ipsilateral (left) secondary somatosensory cortex
regions_of_interest = ["rPSC_2", "rPSC_1", "rPSC_3b", "rSII_TR50_right", "rSII_TR50_left"]

## Plotting
for step in steps:
    if step == "plot_rdms":
        for region in regions_of_interest:
            rdm_path = os.path.join(resultpath, region, 'rdm')
            for index, subject in enumerate(subjects):
                # read in rdm as numpy array
                stimulation_rdm = np.genfromtxt(rdm_path + "/stimulation_rdm_euclidean_" + subject + ".txt", delimiter = ',')[0:-1]
                imagery_rdm = np.genfromtxt(rdm_path + "/imagery_rdm_euclidean_" + subject + ".txt", delimiter = ',')[0:-1]
                
                # plot heatmaps
                stimulation_figure = plot_rdm(stimulation_rdm, subject, stimulation_conditions)

                # plt.show()
                # save figure as jpg file
                stimulation_figure_filename = os.path.join(rdm_path, "stimulation_rdm_euclidean_" + subject + ".jpg")
                stimulation_figure.savefig(stimulation_figure_filename)
                plt.close()

                imagery_figure = plot_rdm(imagery_rdm, subject, imagery_conditions)

                # plt.show()
                # save figure as jpg file
                imagery_figure_filename = os.path.join(rdm_path, "imagery_rdm_euclidean" + subject + ".jpg")
                imagery_figure.savefig(imagery_figure_filename)
                plt.close()
    
    elif step == "plot_rdms_all_conds":
        for region in regions_of_interest:
            rdm_path = os.path.join(resultpath, region, 'rdm')
            for index, subject in enumerate(subjects):
                # read in rdm as numpy array
                all_rdm = np.genfromtxt(rdm_path + "/all_rdm_euclidean_" + subject + ".txt", delimiter = ',')
                
                # plot heatmaps
                all_rdm_figure = plot_rdm(all_rdm, subject, all_conditions)

                # plt.show()
                # save figure as jpg file
                all_rdm_figure_filename = os.path.join(rdm_path, "all_rdm_euclidean_" + subject + ".jpg")
                all_rdm_figure.savefig(all_rdm_figure_filename)
                plt.close()
    
    elif step == "plot_rdm_comparison":
        similiarity_values_all_regions = np.empty(len(subjects)*len(regions_of_interest))
        for index,region in enumerate(regions_of_interest):
            rsa_path = os.path.join(resultpath, region, 'rsa', 'stim_imag_rsa_corr.txt')
            # read in similiarity values for region
            similiarity_values = np.genfromtxt(rsa_path, delimiter=',')[0:-1]
            mean_similiarity_value = np.mean(similiarity_values)
            similiarity_values_all_regions[(index*len(subjects)):((index*len(subjects)) + len(subjects))] = similiarity_values[0:]

        similiarity_plot_data = pd.DataFrame({'ROIs': np.repeat([1, 2, 3, 4, 5], 10),
                            'Similiarity': similiarity_values_all_regions})
        
        similiarity_figure, similiarity_rois_axes = plt.subplots()
        similiarity_rois_axes = sns.barplot(data=similiarity_plot_data, 
                                            x="ROIs", y="Similiarity", 
                                            errorbar='ci')
        similiarity_rois_axes.set_title('Similiarity of Stimulation and Imagery RDMs across regions of interest')
        region_labels = ["right BA2", "right BA1", "right BA3b", "right SII", "left SII"]
        xpos = np.arange(len(region_labels))
        similiarity_rois_axes.set_xticks(xpos, labels=region_labels)
        similiarity_rois_axes.set_ylabel('Similiarity (r) of Stimulation and Imagery RDMs')
        
        # plt.show()
        # save figure as jpg file
        similiarity_figure_filename = os.path.join(resultpath, "anova", "similiarity_stim_imag_across_rois.jpg")
        similiarity_figure.savefig(similiarity_figure_filename)
        plt.close()

    elif step == "plot_rsm_cross_validated":
        similiarity_values_all_regions = np.empty(len(subjects)*len(regions_of_interest))
        for index,region in enumerate(regions_of_interest):
            rsm_path = os.path.join(resultpath, region, 'cross_validation', 'cross_validation_corr_avg.txt')
            # read in similiarity values for region
            similiarity_values = np.genfromtxt(rsm_path, delimiter=',')[0:-1]
            mean_similiarity_value = np.mean(similiarity_values)
            similiarity_values_all_regions[(index*len(subjects)):((index*len(subjects)) + len(subjects))] = similiarity_values[0:]

        similiarity_plot_data = pd.DataFrame({'ROIs': np.repeat([1, 2, 3, 4, 5], 10),
                            'Similiarity': similiarity_values_all_regions})
        
        similiarity_figure, similiarity_rois_axes = plt.subplots()
        similiarity_rois_axes = sns.barplot(data=similiarity_plot_data, 
                                            x="ROIs", y="Similiarity", 
                                            errorbar='ci')
        similiarity_rois_axes.set_title('Cross-validated Similiarity of Stim and Imag RDMs across ROIs')
        region_labels = ["right BA2", "right BA1", "right BA3b", "right SII", "left SII"]
        xpos = np.arange(len(region_labels))
        similiarity_rois_axes.set_xticks(xpos, labels=region_labels)
        similiarity_rois_axes.set_ylabel('Cross-validated Similiarity (r) of Stim and Imag RDMs')
        
        # plt.show()
        # save figure as jpg file
        similiarity_figure_filename = os.path.join(resultpath, "anova", "cross_validated_similiarity_stim_imag_across_rois.jpg")
        similiarity_figure.savefig(similiarity_figure_filename)
        plt.close()
