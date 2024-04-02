import os
import shap
import joblib
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
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, normalize

# Function to load the trained Random Forest model
def load_RF_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print("Error loading Random Forest model:", e)
        return None

# Function to make predictions using the Random Forest model
def predict_with_RF(model, data):
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        print("Error predicting with Random Forest model:", e)
        return None

# Beautify the plots
def beautify_plot(ax):
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def preprocessing_data(raw_data_file):
    try:
        # set sampling frequency to 128 Hz
        sampling_freq = 128
        # create MNE info object
        info = mne.create_info(32, sfreq=sampling_freq)

        raw = mne.io.read_raw_bdf(raw_data_file, preload=True)

        # bandpass frequency filter
        raw_data = raw.filter(l_freq=4, h_freq=45, fir_design='firwin', l_trans_bandwidth='auto', filter_length='auto')

        ica = mne.preprocessing.ICA(n_components=20, random_state=0)
        ica.fit(raw)
        raw_data = ica.apply(raw)

        # Convert to array
        raw_data_array = raw_data.get_data()

        print("Shape of raw_data_array:", raw_data_array.shape)

        print("EEG data shape:", raw_data_array.shape)
        print(raw_data_array)

        return raw_data_array

    except Exception as e:
        print("An error occurred:", e)
        return None

def create_circumplex_model(dataframe):
    try:
        # Set the overall shape and axes
        fig, ax = plt.subplots(figsize=(12, 8))

        # Setting a limit for axes
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(-4.6, 4.6)

        # Setting points and parameters for corresponding emotions
        for index, row in dataframe.iterrows():
            valence = row['Valence']
            arousal = row['Arousal']

            # Correlation of valence and excitation values with coordinates in the model
            x = valence
            y = arousal
            color = 'white'
            marker = "x"
            label = "neutral"
            if y > 0 and x > 0:
                color = 'red'
                marker = "*"
                label = "happy"
            elif y > 0 > x:
                color = 'black'
                marker = "v"
                label = "angry"
            elif y < 0 < x:
                color = 'blue'
                marker = "D"
                label = "fear"
            elif y < 0 and x < 0:
                color = 'green'
                marker = "."
                label = "sad"

            ax.scatter(x, y, s=11, color=color, label=label, alpha=0.5, marker=marker)

        ax.set_xlabel('Valence')
        ax.set_ylabel('Arousal')
        beautify_plot(ax)

        legend_elements = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=8, label='Happy'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=8, label='Angry'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=8, label='Calm'),
            plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=8, label='Sad')
        ]
        ax.legend(handles=legend_elements, loc="lower left")
        plt.show()

    except Exception as e:
        print("Error in create_circumplex_model:", e)


def emotional_labeling(arousal, valence):
    emotions_set = []
    arousal = arousal - 4.5
    valence = valence - 4.5
    for index in range(arousal.size):
        if arousal[index] > 0 and valence[index] > 0:
            # happy
            emotions_set.append(1)
        elif arousal[index] > 0 > valence[index]:
            # angry
            emotions_set.append(2)
        elif arousal[index] < 0 < valence[index]:
            # clam
            emotions_set.append(3)
        elif arousal[index] < 0 and valence[index] < 0:
            # sad
            emotions_set.append(4)
        else:
            emotions_set.append(0)
    return emotions_set


def read_preprocessed_data(file_path):
    try:
        print("Allocation of initial dataframes for EEG data and features:\n")

        data = []
        labels = []

        current_file = spio.loadmat(file_path)

        keys = [key for key, values in current_file.items() if
                key != '__header__' and key != '__version__' and key != '__globals__']

        labels.extend(current_file[keys[0]])  # Using extend instead of append
        data.extend(current_file[keys[1]])  # Using extend instead of append

        labels = np.array(labels)
        data = np.array(data)

        print("Before reshaping:")
        print("Labels shape:", labels.shape)
        print("Data shape:", data.shape)

        labels = labels.reshape(-1, 4)  # Reshape to (1280, 4)
        data = data.reshape(-1, 40, 8064)  # Reshape to (1280, 40, 8064)

        print("After reshaping:")
        print("Labels shape:", labels.shape)
        print("Data shape:", data.shape)

        print("labels - ", labels)
        print("data - ", data)

        return labels, data

    except Exception as e:
        print("Error in read_preprocessed_data:", e)
        return None, None



def feature_extraction(labels, data):
    print("Feature extraction from EEG data based on the power range of each signal:\n")
    eeg_data = data[:, :32, :]
    labels = labels[:, :2]
    print("Labels shape:", labels.shape)
    print("EEG data shape:", eeg_data.shape)

    eeg_band = []
    for i in range(len(eeg_data)):
        for j in range(len(eeg_data[0])):
            eeg_band.append(get_band_power(i, j, "theta", eeg_data))
            eeg_band.append(get_band_power(i, j, "alpha", eeg_data))
            eeg_band.append(get_band_power(i, j, "beta", eeg_data))
            eeg_band.append(get_band_power(i, j, "gamma", eeg_data))

    eeg_band = np.array(eeg_band)
    print("eeg_band size:", eeg_band.size)

    # Ensure the number of samples matches the labels
    num_samples = labels.shape[0]
    expected_size = num_samples * 128
    if eeg_band.size != expected_size:
        print("Error: Size mismatch. Expected size:", expected_size, "Actual size:", eeg_band.size)
        return None

    eeg_band = eeg_band.reshape((num_samples, 128))
    print("After reshaping:")
    print("EEG band shape:", eeg_band.shape)

    # Saving the eeg_band array
    np.save("Deap/Deap dataset/New_PreProcessed_Python/eeg_band.npy", eeg_band)

    return eeg_band

def create_labels(eeg_band_data_):
    print("Labeling arousal, valence training set with appropriate emotions :\n")
    data_with_labels = pd.DataFrame({'Valence': eeg_band_data_[:, 0] - 4.5, 'Arousal': eeg_band_data_[:, 1] - 4.5,
                                     'Emotion': emotional_labeling(eeg_band_data_[:, 0], eeg_band_data_[:, 1])})
    data_with_labels.info()
    data_with_labels.describe()
    df_label_ratings = pd.DataFrame({'Valence': eeg_band_data_[:, 0], 'Arousal': eeg_band_data_[:, 1]})
    # Let's plot the first 40 rows of data
    df_label_ratings.iloc[0:40].plot(style=['o', 'rx'])
    np.save("Deap/Deap dataset/create_labels/labels.npy", data_with_labels)

    create_circumplex_model(data_with_labels)

    return data_with_labels

def bandpower(data, sf, band):
    sns.set(font_scale=1.2)

    band = np.asarray(band)
    # Determine the lower and upper limits of the delta
    low, high = band

    # Determining window length (4 seconds)
    window = (2 / low) * sf
    frequency, psd = signal.welch(data, sf, nperseg=window)

    # Frequency resolution
    freq_res = frequency[1] - frequency[0]

    # Intersecting values in the frequency vector
    idx_band = np.logical_and(frequency >= low, frequency <= high)

    band_power = simps(psd[idx_band], dx=freq_res)
    return band_power

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

def get_band_power(people, channel, band, eeg_data):

    bd = (0, 0)
    if band == "theta":
        bd = (4, 7)
    elif band == "alpha":
        bd = (8, 13)
    elif band == "beta":
        bd = (14, 30)
    elif band == "gamma":
        bd = (31, 50)
    return bandpower(eeg_data[people, channel], 128., bd)

def map_emotion_label(emotion_label):
    if emotion_label == 1:
        return "HAPPY"
    elif emotion_label == 2:
        return "ANGER"
    elif emotion_label == 3:
        return "CALM"
    elif emotion_label == 4:
        return "SAD"
    else:
        return "UNKNOWN"


# Function to explain predictions using SHAP
def explain_predictions(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")


# Function to analyze health implications based on predicted emotions
def analyze_health_implications(predicted_emotion):
    message = ""
    if predicted_emotion == "HAPPY":
        message = " Suggestions From Emotion X - : Predicted emotion is good for health. Maintain a healthy lifestyle to sustain positive emotions."
    elif predicted_emotion == "CALM":
        message = " Suggestions From Emotion X - : Predicted emotion is good for health. Practice relaxation techniques to further enhance calmness."
    elif predicted_emotion == "ANGER":
        message = " Suggestions From Emotion X - : Predicted emotion may have adverse effects on health. Practice anger management techniques, seek professional help if needed."
    elif predicted_emotion == "SAD":
        message = " Suggestions From Emotion X - : Predicted emotion may have adverse effects on health. Seek social support, engage in activities that bring joy, consider therapy if feelings persist."
    else:
        message = " >>> Emotion X - : Predicted emotion does not have defined in this EmotionX System. <<< "

    return message


if __name__ == '__main__':
    input_mat_file = "Deap/Deap dataset/data_preprocessed_matlab/s01.mat"
    labels_for_feature, data_for_feature = read_preprocessed_data(input_mat_file)
    eeg_band_data = feature_extraction(labels_for_feature, data_for_feature)
    labels_for_classification = create_labels(labels_for_feature)

    # classification with random_forest classification

    # Load saved models
    RF_model = load_RF_model('Model.pkl')

    # Make predictions
    RF_predictions = predict_with_RF(RF_model, eeg_band_data)

    # Map numerical labels to emotion types
    predicted_emotion_types = [map_emotion_label(label) for label in RF_predictions]

    # Return the predicted emotion type
    common_emotion = max(set(predicted_emotion_types), key=predicted_emotion_types.count)
    print("Predicted emotion type:", common_emotion)

    # Analyze health implications of predicted emotions
    suggestion = analyze_health_implications(common_emotion)

    print("Classification Report:")
    print(suggestion)