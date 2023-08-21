# script with functions for the main RSA script (DO NOT RUN THIS SCRIPT)
# both scripts need to be in the same directory in order to be able to run the rsa.py script

import os
import numpy as np
import rsatoolbox
import rsatoolbox.data as rsd
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


############################# FUNCTIONS FOR FORMATTING ############################## 

# our data is in the following format:
# 6 runs * 11 regressors + 6 constants = 72 beta files per participant
# regressors:       1: Stim Press
#                   2: Stim Flutt
#                   3: Stim Vibro
#                   4: Imag Press (only "successful" trials)
#                   5: Imag Flutt (only "successful" trials)
#                   6: Imag Vibro (only "successful" trials)
#                   7: Null 1
#                   8: Null 2
#                   9: pre Cue
#                   10: button press
#                   11: all the remaining (bad) Imag trials


# format_data_for_subject() takes a datapath, the selected runs,
# the selected conditions, and all conditions as input,
# removes unnecessary regressors, loads data and sorts beta values into conditions
# returns formatted data for the subject
def format_data_for_subject(datapath: str,
                            selected_runs: list[str],
                            selected_conditions: list[str],
                            all_conditions: list[str]) -> np.ndarray:
    # remove regressors that we don't need (all except 6 conditions)
    filtered_beta_files = remove_regressors(datapath)
    # load only relevant data
    betas_sub = load_data(filtered_beta_files, datapath)
    # sort beta values into 6 conditions
    return sort_data_into_conditions(selected_conditions, all_conditions,
                                      selected_runs, betas_sub)


# remove_regressors() takes a datapath as input 
# returns a list with the beta files that we need (only regressors 1-6, see above)
def remove_regressors(datapath: str) -> list[str]:
    # lists all beta files for the subject, sorted by name
    beta_files_subject = sorted(glob.glob(os.path.join(datapath, 'beta*.nii')))
    # only get the relevant beta files
    # (6 files per condition, skipping 5 unused files, 6 times in total)
    filtered_beta_files = []
    for index in range(0, 56, 11):
        filtered_beta_files += beta_files_subject[index:index+6]

    if len(filtered_beta_files) != 36:
        raise ValueError("Number of beta files is not 36!")
    
    return filtered_beta_files


# load_data() takes a list with the filtered beta files and a datapath as input
# returns a list with the loaded betas
def load_data(filtered_beta_files: list[str],
              datapath: str) -> list[int]:
    betas_subject = []
    for beta_file in filtered_beta_files:
        # load nifti files
        file_path = os.path.join(datapath, beta_file)
        beta = nib.load(file_path)
        # get data of nifti files
        beta_data = beta.get_fdata()
        # add data of file to subject list
        betas_subject += [beta_data]
    # return list with relevant beta files for subject
    return betas_subject


# sort_data_into_conditions() takes the selected conditions, all conditions, 
# a list of the selected runs and a list of unsorted beta files as input
# returns a 5D array with the following dimensions:
#       1st dimension: 79 voxels
#       2nd dimension: 95 voxels
#       3rd dimension: 79 voxels
#       4th dimension: 6 conditions/stimulus types
#       5th dimension: selected runs (1-6)
def sort_data_into_conditions(selected_conditions: list[str],
                                   all_conditions: list[str],
                                   selected_runs: list[str],
                                    betas_unsorted: list[int]) -> np.ndarray:
    # transform list of selected runs to list of integers
    selected_runs_int = [int(run)-1 for run in selected_runs]
    # initiate empty 5D array to fill
    separated_conditions_runs = np.empty((79, 95, 79, len(selected_conditions),
                                           len(selected_runs)))
    for index, condition in enumerate(selected_conditions):
        # get all betas for the condition
        # get index for condition (from all_conditions)
        num = all_conditions.index(condition)
        # add all runs of stimulus together
        stimulus = np.stack(betas_unsorted[num::6], axis=-1)
        # select only the runs that we need
        stimulus_selected_runs = stimulus[:, :, :, selected_runs_int]
        # add to main array
        separated_conditions_runs[:, :, :, index, :] = stimulus_selected_runs
    return separated_conditions_runs


# average_over_runs() takes the already formatted data (6D array) as input
# and returns a 5D array of the data averaged over the selected runs
def average_over_runs(formatted_data: np.ndarray) -> np.ndarray:
    averaged_data = np.mean(formatted_data, axis=4)
    return averaged_data


# get_voxels_from_region_of_interest() takes a region and a datapath as input
# loads a nifti file which defines that region and returns it as np.ndarray
def get_voxels_from_region_of_interest(region_of_interest: str,
                                       datapath: str) -> np.ndarray:
    folder_path = os.path.join(
        datapath, "rois", "*" + region_of_interest + "*.nii")
    all_files_path = glob.glob(folder_path)
    # check if there is only one file
    if len(all_files_path) != 1:
        raise ValueError("There is not exactly one region file!")
    file_path = all_files_path[0]
    # get data of nifti file
    region_data = nib.load(file_path).get_fdata()
    return region_data
    

# rearrange_array() takes an np.ndarray that defines a region of interest and
# a np.ndarray with the data as inputs
# indexs the data with our region mask to only extract the voxels of that region
# combines the x, y, z voxel coordinates (first three dimensions) into a flat vector 
# and rearranges array dimensions
# 1 (conditions) -> 0, 0 (voxels) -> 1, 2 (participants) -> 2
# returns a 3D data array with the dimensions:
#       1st dimension: 6 conditions/stimulus types
#       2nd dimension: roi voxels
#       3rd dimension: 10 participants
def rearrange_array(region_data: np.ndarray,
                    formatted_data: np.ndarray) -> np.ndarray:
    # get all the indices of region data which are non-zero (that define the region)
    # and convert indices to flat index
    region_indices_flat = np.ravel_multi_index(np.nonzero(region_data),
                                               formatted_data.shape[:3])
    # access only the indexed voxels of our big conditions array
    voxels_of_selected_conditions = formatted_data.reshape(
        -1, *formatted_data.shape[3:])[region_indices_flat]
    # rearrange array so it fits the toolbox data structure
    # 1 (conditions) -> 0, 0 (voxels) -> 1, 2 (participants) -> 2
    selected_conditions_region_only = np.transpose(
        voxels_of_selected_conditions, (1, 0, 2))
    return selected_conditions_region_only


############################# FUNCTIONS FOR RSA ############################## 

# create_rsa_dataset() takes the data from a region, the number of subjects
# and a condition key as input
# returns a RSAToolbox object using the RSAToolbox by Schütt et al., 2019
# with the following attributes:
#       data.measurements: 634 voxel values for 6 conditions
#       data.descriptors: subj no
#       data.obs_descriptors: cond no
#       data.channel_descriptors: vox no
def create_rsa_datasets(data_from_region: np.ndarray,
                        subject_count: int,
                        condition_key: str) -> list[rsd.Dataset]:
    voxel_count = data_from_region.shape[1]
    condition_description = {condition_key: np.array([condition_key + str(c + 1)
                                                    for c in np.arange(data_from_region.shape[0])])}
    voxel_description = {'voxels': np.array(['voxel_' + str(x + 1)
                                             for x in np.arange(voxel_count)])}
    rsa_data = [] # list of dataset objects
    for subject in np.arange(subject_count):
        descriptors = {'subjects': subject+1}
        # append the dataset object to the data list
        rsa_data += [rsd.Dataset(measurements=data_from_region[:,:,subject],
                            descriptors=descriptors,
                            obs_descriptors=condition_description,
                            channel_descriptors=voxel_description
                            )]
    return rsa_data


# show_debug_for_rdm() takes the data of a representational dissimiliarity matrix as input
# prints it and plots a figure to check if everything went alright
def show_debug_for_rdm(rdm_data: rsatoolbox.rdm.RDMs):
    print(rdm_data)
    figure = rsatoolbox.vis.show_rdm(
        rdm_data, show_colorbar='figure')
    figure.show()


# save_rdm_results() takes a resultpath, a region, a condition, a method
# and the data of a representational dissimiliarity matrix as input
# saves the matrix
def save_rdm_results(resultpath: str,
                     region: str,
                     condition: str,
                     method: str,
                     rdm_data: np.ndarray,
                     subject: str):
    # make a new directory for the region and rdm
    rdm_path = os.path.join(
        resultpath, region, "rdm")
    if os.path.exists(rdm_path) == False:
        os.makedirs(rdm_path)

    # save matrix as text file
    matrix_filename = os.path.join(rdm_path, condition + 
                                   "_rdm_" + method + "_" + subject + ".txt")
    np.savetxt(matrix_filename, rdm_data, delimiter=',')


# save_rsa_results() takes a resultpath, a region, a condition, a method
# and the data of a representational similiarity analysis as input
# saves the rsa results to the specified directory
def save_rsa_results(resultpath: str,
                     region: str,
                     condition: str,
                     method: str,
                     rsa_data):
    # make a new directory for the region and rsa
    rsa_path = os.path.join(
        resultpath, region, "rsa")
    if os.path.exists(rsa_path) == False:
        os.makedirs(rsa_path)

    # save rsa matrix as text file
    filename = os.path.join(rsa_path, condition + "_rsa_" + method + ".txt")
    if os.path.exists(filename) == True:
        os.remove(filename)
        
    if method == 'ttest':
        file = open(filename, 'w')
        file.write(str(rsa_data))
        file.close()
    else:
        file = open(filename, 'a')
        for element in rsa_data:
            file.write(str(element) + ",")
        file.close()


######################### FUNCTIONS FOR VISUALIZATION ########################## 

# plot_rdm() takes a rdm, a subject and a list of conditions as input
# plots heatmap of the rdm using python seaborn library
def plot_rdm(rdm: np.ndarray,
             subject: str,
             conditions: list[str]):
    figure, axes = plt.subplots()
    axes = sns.heatmap(rdm,
            cbar_kws = {'label':'Dissimiliarity (Euclidean)'},
            cmap='viridis')
    axes.xaxis.tick_top()
    axes.set_xticklabels(conditions, rotation=0, fontsize=8)
    axes.set_yticklabels(conditions, rotation=0, fontsize=8)
    axes.set_title('subject ' + subject)
    return figure