import os
import pickle


import mne
import pandas as pd
import scipy
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


import matplotlib.pyplot as plt

# !pip install scikit-learn==0.20.3

# !pip install mne==0.22.0
# from mne.time_frequency import psd_welch

# from mne.time_frequency import psd_welch

# from mne.time_frequency import psd_welch

# !pip install fooof

# %matplotlib inline
import mne
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs, corrmap)
# from mne.time_frequency import psd_welch

# Import the appropriate function for time-frequency analysis from mne.time_frequency
# from mne.time_frequency import some_alternative_function

from mne.decoding import cross_val_multiscore

import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

from scipy import signal
from scipy.signal import welch
from scipy.integrate import simps
from scipy.integrate import simpson
from scipy.stats import f_oneway

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve


#
# def preprocessing_data(input_file, output_folder):
#     # Set sampling frequency to 128 Hz
#     sampling_freq = 128
#     # Create MNE info object
#     info = mne.create_info(32, sfreq=sampling_freq)
#
#     # Read raw data from the BDF file
#     raw = mne.io.read_raw_bdf(input_file, preload=True)
#
#     # Plot the raw data (optional)
#     raw.plot()
#
#     # Print information about the raw data
#     print(raw.info)
#
#     # Apply bandpass frequency filter
#     raw_data = raw.filter(l_freq=4, h_freq=45, fir_design='firwin', l_trans_bandwidth='auto', filter_length='auto')
#
#     # Apply Independent Component Analysis (ICA)
#     ica = mne.preprocessing.ICA(n_components=20, random_state=0)
#     ica.fit(raw)
#     raw_data = ica.apply(raw)
#
#     # Plot data again after removing bad channels and interpolating
#     raw_data.compute_psd(fmax=50).plot(picks="data", exclude="bads")
#     raw_data.plot(block=True)
#
#     # # Load additional data (just an example, replace with your actual data loading code)
#     # additional_data = scipy.io.loadmat('path_to_additional_data.mat')
#     # print(additional_data)
#
#     # Get the subject index from the input file name
#     subject_index = os.path.splitext(os.path.basename(input_file))[0][1:]
#
#     # Save preprocessed data in the specified output folder as s(i).dat
#     output_file = f"{output_folder}/s{subject_index}.fif"
#     raw_data.save(output_file, overwrite=True)
#
#     return raw_data
#
# # Example usage:
# file = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/data_original/s01.bdf"
# output_folder_path = '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Fif_File/'
# preprocessing_data(file, output_folder_path)
#
#
# def convert_fif_to_dat(input_file, output_file):
#     try:
#         with open(input_file, 'rb') as infile:
#             # Read the content of the .fif file
#             file_content = infile.read()
#
#         with open(output_file, 'wb') as outfile:
#             # Write the content to the .dat file
#             outfile.write(file_content)
#
#         print(f"Conversion from {input_file} to {output_file} successful.")
#
#     except FileNotFoundError:
#         print("File not found. Please check the file paths.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
#
# # Example usage:
# input_fif_file = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Fif_File/s01.fif"
# output_dat_file = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Python/s01.dat"
#
# convert_fif_to_dat(input_fif_file, output_dat_file)
#

# Function to load data from each participant file

def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p

# Load only 22/32 participants with frontal videos recorded
files = []
for n in range(1, 23):
    s = ''
    if n < 10:
        s += '0'
    s += str(n)
    files.append(s)
print(files)

print()


# 22x40 = 880 trials for 22 participants
labels = []
data = []

for i in files:
  filename = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/data_preprocessed_python/s" +i + ".dat"
  trial = read_eeg_signal_from_file(filename)
  labels.append(trial['labels'])
  data.append(trial['data'])

# Re-shape arrays into desired shapes
labels = np.array(labels)
labels = labels.flatten()
labels = labels.reshape(880, 4)

data = np.array(data)
data = data.flatten()
data = data.reshape(880, 40, 8064)

# Double-check the new arrays
print("Labels: ", labels.shape) # trial x label
print("Data: ", data.shape) # trial x channel x data
print()


# Only extract Valence and Arousal ratings
df_label_ratings = pd.DataFrame({'Valence': labels[:,0], 'Arousal': labels[:,1]})

print(df_label_ratings.describe())

print()

print(df_label_ratings.head(15))

df_label_ratings.iloc[0:40].plot(style=['o','rx'])

plt.show()

print()

# Check positive/negative cases
# The combinations of Valence and Arousal can be converted to emotional states: High Arousal Positive Valence (Excited, Happy), Low Arousal Positive Valence (Calm, Relaxed), High Arousal Negative Valence (Angry, Nervous), and Low Arousal Negative Valence (Sad, Bored).

# High Arousal Positive Valence,Dominance dataset
df_hahv = df_label_ratings[(df_label_ratings['Valence'] >= np.median(labels[:,0])) & (df_label_ratings['Arousal'] >= np.median(labels[:,1]))]

# Low Arousal Positive Valence,Dominance dataset
df_lahv = df_label_ratings[(df_label_ratings['Valence'] >= np.median(labels[:,0])) & (df_label_ratings['Arousal'] < np.median(labels[:,1]))]

# High Arousal Negative Valence,Dominance dataset
df_halv = df_label_ratings[(df_label_ratings['Valence'] < np.median(labels[:,0])) & (df_label_ratings['Arousal'] >= np.median(labels[:,1]))]

# Low Arousal Negative Valence,Dominance dataset
df_lalv = df_label_ratings[(df_label_ratings['Valence'] < np.median(labels[:,0])) & (df_label_ratings['Arousal'] < np.median(labels[:,1]))]

# #Low Arousal Negative Valence dataset
# #df_lalv = df_label_ratings[(df_label_ratings['Valence'] < np.median(labels[:,0])) & (df_label_ratings['Arousal'] < np.median(labels[:,1])) & (df_label_ratings['Dominance'] < np.median(labels[:,2]))]

# Check number of trials per each group

print("Positive Valence (+) :", str(len(df_hahv) + len(df_lahv)))

print("Negative Valence (-) :", str(len(df_halv) + len(df_lalv)))

print("High Arousal (H) :", str(len(df_hahv) + len(df_halv)))

print("Low Arousal  (L) :", str(len(df_lahv) + len(df_lalv)))

print()

# Check number of trials per each group
print("High Arousal Positive Valence (+) :", str(len(df_hahv)))
print("Low Arousal Positive Valence  (-) :", str(len(df_lahv)))
print("High Arousal Negative Valence (H) :", str(len(df_halv)))
print("Low Arousal Negative Valence  (L) :", str(len(df_lalv)))

print()


# Get mean and std of each group

print("High Arousal High Valence")
print("Valence:", "Mean", np.round(df_hahv['Valence'].mean(),2), "STD", np.round(df_hahv['Valence'].std(),2))
print("Arousal:", "Mean", np.round(df_hahv['Arousal'].mean(),2), "STD", np.round(df_hahv['Arousal'].std(),2))

print()
print("Low Arousal High Valence:")
print("Valence:", "Mean", np.round(df_lahv['Valence'].mean(),2), "STD", np.round(df_lahv['Valence'].std(),2))
print("Arousal:", "Mean", np.round(df_lahv['Arousal'].mean(),2), "STD", np.round(df_lahv['Arousal'].std(),2))

print()
print("High Arousal Low Valence:")
print("Valence:", "Mean", np.round(df_halv['Valence'].mean(),2), "STD", np.round(df_halv['Valence'].std(),2))
print("Arousal:", "Mean", np.round(df_halv['Arousal'].mean(),2), "STD", np.round(df_halv['Arousal'].std(),2))

print()
print("High Arousal Low Valence:")
print("Valence:", "Mean", np.round(df_lalv['Valence'].mean(),2), "STD", np.round(df_lalv['Valence'].std(),2))
print("Arousal:", "Mean", np.round(df_lalv['Arousal'].mean(),2), "STD", np.round(df_lalv['Arousal'].std(),2))

print()

fig, axs = plt.subplots(1, 2, figsize=(12,4))

axs[0].set_title("Valence")
axs[0].set_ylim(1, 9)
axs[0].boxplot([df_hahv['Valence'], df_lahv['Valence'], df_halv['Valence'], df_lalv['Valence']], labels=['HAHV','LAHV','HALV', 'LALV'])

axs[1].set_title("Arousal")
axs[1].set_ylim(1, 9)
axs[1].boxplot([df_hahv['Arousal'], df_lahv['Arousal'], df_halv['Arousal'], df_lalv['Arousal']], labels=['HAHV','LAHV','HALV', 'LALV'])

plt.show()

# Valence , Arousal and Dominance ratings per group
fig, axs = plt.subplots(2, 2, figsize=(12,8))

axs[0,0].set_title("HAHV")
axs[0,0].set_ylim(1, 9)
axs[0,0].boxplot([df_hahv['Valence'], df_hahv['Arousal']], labels=['Valence','Arousal'])

axs[0,1].set_title("LAHV")
axs[0,1].set_ylim(1, 9)
axs[0,1].boxplot([df_lahv['Valence'], df_lahv['Arousal']], labels=['Valence','Arousal'])

axs[1,0].set_title("HALV")
axs[1,0].set_ylim(1, 9)
axs[1,0].boxplot([df_halv['Valence'], df_halv['Arousal']], labels=['Valence','Arousal'])

axs[1,1].set_title("LALV")
axs[1,1].set_ylim(1, 9)
axs[1,1].boxplot([df_lalv['Valence'], df_lalv['Arousal']], labels=['Valence','Arousal'])

plt.show()

# encoding

# Function to check if each trial has positive or negative valence
def positive_valence(trial):
    return 1 if labels[trial,0] >= np.median(labels[:,0]) else 0
# Function to check if each trial has high or low arousal
def high_arousal(trial):
    return 1 if labels[trial,1] >= np.median(labels[:,1]) else 0

# Convert all ratings to boolean values
labels_encoded = []

for i in range (len(labels)):
  labels_encoded.append([positive_valence(i), high_arousal(i)])
labels_encoded = np.reshape(labels_encoded, (880, 2))
df_labels = pd.DataFrame(data=labels_encoded, columns=["Positive Valence", "High Arousal"])
print(df_labels.describe())

print()

# Dataset with only Valence column
df_valence = df_labels['Positive Valence']

# Dataset with only Arousal column
df_arousal = df_labels['High Arousal']

print()

#
# Separate EEG and non-EEG data
# The dataset includes 32 EEG channels and 8 peripheral physiological channels. The peripheral signals include: electrooculogram (EOG), electromyograms (EMG) of Zygomaticus and Trapezius muscles, GSR, respiration amplitude, blood volume by plethysmograph, skin temperature.

eeg_channels = np.array(["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"])
non_eeg_channels = np.array(["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"])

eeg_data = []
for i in range (len(data)):
  for j in range (len(eeg_channels)):
    eeg_data.append(data[i,j])
eeg_data = np.reshape(eeg_data, (len(data), len(eeg_channels), len(data[0,0])))
print(eeg_data.shape)

print()

non_eeg_data = []
for i in range (len(data)):
  for j in range (32,len(data[0])):
    non_eeg_data.append(data[i,j])
non_eeg_data = np.reshape(non_eeg_data, (len(data), len(non_eeg_channels), len(data[0,0])))
print(non_eeg_data.shape)

print()

# Functions to get band power values ------------------------------

"""Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
def bandpower(data, sf, band, window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    # bp = simps(psd[idx_band], dx=freq_res)
    bp = scipy.integrate.simpson(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def get_band_power(trial, channel, band):
  bd = (0, 0)

  if (band == "delta"):
    bd = (1, 3)
  elif (band == "theta"):  # drownsiness, emotional connection, intuition, creativity
    bd = (4, 7)
  elif (band == "alpha"):  # reflection, relaxation
    bd = (8, 13)
  elif (band == "beta"):  # concentration, problem solving, memory
    bd = (14, 30)
  elif (band == "gamma"):  # cognition, perception, learning, multi-tasking
    bd = (31, 50)

  return bandpower(eeg_data[trial, channel], 128, bd)


print(get_band_power(0, 31, "delta"))
print(get_band_power(0, 31, "theta"))
print(get_band_power(0, 31, "alpha"))
print(get_band_power(0, 31, "beta"))
print(get_band_power(0, 31, "gamma"))

info = mne.create_info(32, sfreq=128)
print(info)


# Process new datasets with 6 EEG regions and 4 band power values

# Transform 880 x 32 x 8064 => 880 x 128
eeg_band_arr = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_band_arr.append(get_band_power(i,j,"theta"))
    eeg_band_arr.append(get_band_power(i,j,"alpha"))
    eeg_band_arr.append(get_band_power(i,j,"beta"))
    eeg_band_arr.append(get_band_power(i,j,"gamma"))
eeg_band_arr = np.reshape(eeg_band_arr, (880, 128))

# EEG channels are divided into 6 groups, according to their
# cerebral spatial locations.
# We combine the temmporal-left and temporal-right channels into the left and right groups.
# The frontal group only includes the fronto-central channels.
# All centro-parietal channels are included in the central group.

left = np.array(["Fp1", "AF3", "F7", "FC5", "T7"])
right = np.array(["Fp2", "AF4", "F8", "FC6", "T8"])
frontal = np.array(["F3", "FC1", "Fz", "F4", "FC2"])
parietal = np.array(["P3", "P7", "Pz", "P4", "P8"])
occipital = np.array(["O1", "Oz", "O2", "PO3", "PO4"])
central = np.array(["CP5", "CP1", "Cz", "C4", "C3", "CP6", "CP2"])


# Dataframe for Delta power values

# Transform 880 x 32 x 8064 => 880 x 32
eeg_delta = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_delta.append(get_band_power(i,j,"delta"))
eeg_theta = np.reshape(eeg_delta, (880, 32))

df_delta = pd.DataFrame(data = eeg_theta, columns=eeg_channels)
print(df_delta.describe())

# Only print central channels
print(df_delta[frontal].head(5))


# Dataframe for Theta power values

# Transform 880 x 32 x 8064 => 880 x 32
eeg_theta = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_theta.append(get_band_power(i,j,"theta"))
eeg_theta = np.reshape(eeg_theta, (880, 32))

df_theta = pd.DataFrame(data = eeg_theta, columns=eeg_channels)
print(df_theta.describe())

# Only print central channels
print(df_theta[central].head(5))



# Dataframe for Alpha power values

# Transform 880 x 32 x 8064 => 880 x 32
eeg_alpha = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_alpha.append(get_band_power(i,j,"alpha"))
eeg_alpha = np.reshape(eeg_alpha, (880, 32))

df_alpha = pd.DataFrame(data = eeg_alpha, columns=eeg_channels)
print(df_alpha.describe())

# Only print occipital channels
print(df_alpha[occipital].head(5))



# Dataframe for Beta power values

# Transform 880 x 32 x 8064 => 880 x 32
eeg_beta = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_beta.append(get_band_power(i,j,"beta"))
eeg_beta = np.reshape(eeg_beta, (880, 32))

df_beta = pd.DataFrame(data = eeg_beta, columns=eeg_channels)
print(df_beta.describe())

# Only print frontal channels
print(df_beta[frontal].head(5))



# Dataframe for Gamma power values

# Transform 880 x 32 x 8064 => 880 x 32
eeg_gamma = []
for i in range (len(eeg_data)):
  for j in range (len(eeg_data[0])):
    eeg_gamma.append(get_band_power(i,j,"gamma"))
eeg_gamma = np.reshape(eeg_gamma, (880, 32))

df_gamma = pd.DataFrame(data = eeg_gamma, columns=eeg_channels)
print(df_gamma.describe())

# Only print parietal channels
print(df_gamma[parietal].head(5))


# Train-test split and feature scaling

# Split the data into training/testing sets
def split_train_test(x, y):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
  return x_train, x_test, y_train, y_test

# Feature scaling
def feature_scaling(train, test):
  sc = StandardScaler()
  train = sc.fit_transform(train)
  test = sc.transform(test)
  return train, test

# Cross-validation to select classifier

band_names = np.array(["theta", "alpha", "beta", "gamma"])
channel_names = np.array(["left", "frontal", "right", "central", "parietal", "occipital"])
label_names = np.array(["valence", "arousal"])

# Testing different kernels (linear, sigmoid, rbf, poly) to select the most optimal one
clf_svm = SVC(kernel = 'linear', random_state = 42, probability=True)

# Testing different k (odd) numbers, algorithm (auto, ball_tree, kd_tree) and weight (uniform, distance) to select the most optimal one
clf_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')

# Testing different learning rate (alpha), solver (adam, sgd, lbfgs) and activation (relu, tanh, logistic) to select the most optimal one
clf_mlp = MLPClassifier(solver='adam', activation='tanh', alpha=0.3, max_iter=400)

models = []
models.append(('SVM', clf_svm))
models.append(('k-NN', clf_knn))
models.append(('MLP', clf_mlp))


import time

def cross_validate_clf(df_x, df_y, scoring):
  # Train-test split
  x_train, x_test, y_train, y_test = split_train_test(df_x, df_y)
  # Feature scaling
  x_train, x_test = feature_scaling(x_train, x_test)

  names = []
  means = []
  stds = []
  times = []

  # Apply CV
  for name, model in models:
      start_time = time.time()
      kfold = model_selection.KFold(n_splits=5)
      cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
      t = (time.time() - start_time)
      times.append(t)
      means.append(cv_results.mean())
      stds.append(cv_results.std())
      names.append(name)

  return names, means, stds, times


# Arousal - Accuracy

cross_validate_clf(eeg_band_arr, df_arousal, 'accuracy')

# Arousal - F1

cross_validate_clf(eeg_band_arr, df_arousal, 'f1')

# Valence - Accuracy

cross_validate_clf(eeg_band_arr, df_valence, 'accuracy')

# Valence - F1

cross_validate_clf(eeg_band_arr, df_valence, 'f1')



# Apply classifier
# Functions to run classifiers

def run_clf_cv(band, channel, label, clf):
  if (band == "theta"):
    df_x = df_theta
  elif (band == "alpha"):
    df_x = df_alpha
  elif (band == "beta"):
    df_x = df_beta
  elif (band == "gamma"):
    df_x = df_gamma

  if (channel == "left"):
    df_x = df_x[left]
  elif (channel == "right"):
    df_x = df_x[right]
  elif (channel == "frontal"):
    df_x = df_x[frontal]
  elif (channel == "central"):
    df_x = df_x[central]
  elif (channel == "parietal"):
    df_x = df_x[parietal]
  elif (channel == "occipital"):
    df_x = df_x[occipital]

  df_y = df_arousal if (label == "arousal") else df_valence

  # Train-test split
  x_train, x_test, y_train, y_test = split_train_test(df_x, df_y)

  # Apply CV
  x_for_kfold = np.array(x_train)
  y_for_kfold = np.array(y_train)
  kfold = model_selection.KFold(n_splits=5)

  for i, j in kfold.split(x_for_kfold):
   x_train2, x_test2 = x_for_kfold[i], x_for_kfold[j]
   y_train2, y_test2 = y_for_kfold[i], y_for_kfold[j]

  # Feature scaling
  x_train2, x_test2 = feature_scaling(x_train2, x_test2)

  if (clf == "svm"):
    clf_svm.fit(x_train2, y_train2)
    y_predict = clf_svm.predict(x_test2)
  elif (clf == "knn"):
    clf_knn.fit(x_train2, y_train2)
    y_predict = clf_knn.predict(x_test2)
  elif (clf == "mlp"):
    clf_mlp.fit(x_train2, y_train2)
    y_predict = clf_mlp.predict(x_test2)

  return y_test2, y_predict


def get_accuracy(band, channel, label, clf):
  y_test2, y_predict = run_clf_cv(band, channel, label, clf)
  return np.round(accuracy_score(y_test2, y_predict)*100,2)

def print_accuracy(label, clf):
  arr = []
  for i in range (len(band_names)):
    for j in range (len(channel_names)):
      arr.append(get_accuracy(band_names[i], channel_names[j], label, clf))
  arr = np.reshape(arr, (4,6))
  df = pd.DataFrame(data = arr, index=band_names, columns=channel_names)

  print("Top 3 EEG regions with highest scores")
  print(df.apply(lambda s: s.abs()).max().nlargest(3))
  print()
  print("Top 2 bands with highest scores")
  print(df.apply(lambda s: s.abs()).max(axis=1).nlargest(2))
  print()
  print("EEG region with highest scores per each band")
  print(df.idxmax(axis=1))
  print()
  print("Band with highest scores per each EEG region")
  print(df.idxmax())
  print()
  print(df)

def get_f1(band, channel, label, clf):
    y_test2, y_predict = run_clf_cv(band, channel, label, clf)
    return np.round(f1_score(y_test2, y_predict) * 100, 2)

def print_f1(label, clf):
  arr = []
  for i in range (len(band_names)):
    for j in range (len(channel_names)):
      arr.append(get_f1(band_names[i], channel_names[j], label, clf))
  arr = np.reshape(arr, (4,6))
  df = pd.DataFrame(data = arr, index=band_names, columns=channel_names)

  print("Top 3 EEG regions with highest scores")
  print(df.apply(lambda s: s.abs()).max().nlargest(3))
  print()
  print("Top 2 bands with highest scores")
  print(df.apply(lambda s: s.abs()).max(axis=1).nlargest(2))
  print()
  print("EEG region with highest scores per each band")
  print(df.idxmax(axis=1))
  print()
  print("Band with highest scores per each EEG regions")
  print(df.idxmax())
  print()
  print(df)


def plot_cm(band, channel, label, clf):
  y_test2, y_predict = run_clf_cv(band, channel, label, clf)
  cm = confusion_matrix(y_test2, y_predict)
  print(cm)
  cr = classification_report(y_test2, y_predict)
  print(cr)

  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.colorbar()

  if (label == "valence"):
    classes = df_valence.unique().tolist()
  if (label == "arousal"):
    classes = df_arousal.unique().tolist()

  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], 'd'),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')


# Accuracy results



# Valence
# Only use k-NN in case of Valence after CV
print_accuracy('valence', 'knn')

# Arousal
# Only use MLP in case of Arousal after CV
print_accuracy('arousal', 'mlp')



# F1 score results


# Valence
# Only use k-NN in case of Valence after CV
print_f1('valence', 'knn')


# Arousal
# Only use MLP in case of Arousal after CV
print_f1('arousal', 'mlp')



# Plot results
# Top combinations for Valence

plot_cm('theta', 'central', 'valence', 'knn')
plt.show()

plot_cm('beta', 'left', 'valence', 'knn')
plt.show()

plot_cm('gamma', 'right', 'valence', 'knn')
plt.show()




# Top combinations for Arousal

plot_cm('alpha', 'central', 'arousal', 'mlp')
plt.show()

plot_cm('theta', 'parietal', 'arousal', 'mlp')
plt.show()