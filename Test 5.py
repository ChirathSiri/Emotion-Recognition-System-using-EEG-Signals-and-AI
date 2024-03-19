
import os

import joblib
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
from collections import Counter

def preprocessing_data(raw_data_file):
    try:
        # set sampling frequency to 128 Hz
        sampling_freq = 128
        # create MNE info object
        info = mne.create_info(32, sfreq=sampling_freq)

        raw = mne.io.read_raw_bdf(raw_data_file, preload=True)
        raw.plot()

        # print information about the raw data
        print(raw.info)

        #  bandpass frequency filter
        raw_data = raw.filter(l_freq=4, h_freq=45, fir_design='firwin', l_trans_bandwidth='auto', filter_length='auto')
        print(raw_data.info)
        raw_data.plot(block=True)

        ica = mne.preprocessing.ICA(n_components=20, random_state=0)
        ica.fit(raw)
        raw_data = ica.apply(raw)

        # plot data again after removing bad channels and interpolating
        raw_data.compute_psd(fmax=50).plot(picks="data", exclude="bads")
        raw_data.plot(block=True)



        # data = scipy.io.loadmat('/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/data_preprocessed_matlab/s05.mat')
        # print(data)

        return raw_data

    except Exception as e:
        print("Error in preprocessing_data:", e)
        return None


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

def read_preprocessed_data(directory):
    try:
        print("Allocation of initial dataframes for EEG data and features:\n")

        data = []
        labels = []
        files = os.listdir(directory)
        for file in files:
            current_file = spio.loadmat(directory + file)

            keys = [key for key, values in current_file.items() if
                    key != '__header__' and key != '__version__' and key != '__globals__']

            labels.extend(current_file[keys[0]])  # Using extend instead of append
            data.extend(current_file[keys[1]])    # Using extend instead of append

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
    np.save("/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Python/eeg_band.npy", eeg_band)

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
    np.save("/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Python/eeg_labels", data_with_labels)

    create_circumplex_model(data_with_labels)

    return data_with_labels


def create_circumplex_model(dataframe):
    # Set the overall shape and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Setting a limit for axes
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)
    print(dataframe)

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
    ax.grid(True)

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=8, label='Happy'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=8, label='Angry'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=8, label='Calm'),
        plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=8, label='Sad')
    ]
    ax.legend(handles=legend_elements, loc="lower left")
    plt.show()


def svm_classifier(data, labels):
    print("Support Vector Machine Classifications")
    """
    A function to train an SVM classifier on a data set and perform classification.

    Arguments:
    - data: np.array, a data set of size (1280, 160) with EEG measurements.
    - labels: np.array, class labels of size (1280, 3) in the format [arousal (float), valence (float), emotion (string)].

    Returns:
    - accuracy: float, classification accuracy on test data.
    - report: str, classification report (precision, recall, f1-score) on test data.
    """

    # Retrieving individual class label columns
    emotion = labels[:, 2]

    # Dividing data into training and test sets
    # data - matrix of feature objects and emotion - vector of responses
    X_train, X_test, y_train, y_test = train_test_split(data, emotion, test_size=0.3, random_state=42)

    # Creating and training an SVM classifier
    classifier = SVC(decision_function_shape='ovr')
    classifier.fit(X_train, y_train)
    # Predicting class labels on test data
    predicted = classifier.predict(X_test)
    # Assessment of classification accuracy
    accuracy = accuracy_score(predicted, y_test)
    # Generating a classification report
    report = classification_report(y_test, predicted, zero_division=1)

    # # Convert numerical emotion labels to emotion types
    # final_emotions = [map_emotion_label(labels) for labels in predicted]
    # # train_and_save_model(X_train, y_train, report)
    # print("FINAL EMOTION TYPE:", final_emotions)

    # Assuming map_emotion_label is a function that maps numerical labels to emotion types
    final_emotions = [map_emotion_label(labels) for labels in predicted]

    # Count occurrences of each emotion type
    emotion_counts = Counter(final_emotions)

    # Get the most common emotion type
    most_common_emotion = emotion_counts.most_common(1)[0][0]

    print("SVM Most common emotion type:", most_common_emotion)

    print("Classification accuracy of 4 emotions: " + str(accuracy))
    print(report)

    # Show graph
    plt.show()
    plt.title('SVM Model Score', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(linestyle=':', axis='y')
    a = plt.bar("SVM", accuracy, 0.3, color='red', label='test', align='center')
    plt.show()

    return accuracy


def random_forest_classifier(data, labels):
    print("Classifications using the random_forest method")
    """
   A function for training a Random Forest classifier on a data set and performing classification.

    Arguments:
    - data: np.array, a data set of size (1280, 160) with EEG measurements.
    - labels: np.array, class labels of size (1280, 3) in the format [arousal (float), valence (float), emotion (string)].

    Returns:
    - accuracy: float, classification accuracy on test data.
    - report: str, classification report (precision, recall, f1-score) on test data.
    """

    # Retrieving individual class label columns
    emotion = labels[:, 2]

    # Dividing data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, emotion, test_size=0.3, random_state=42)

    # Creation and training of the Random Forest classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Predicting class labels on test data
    y_pred = classifier.predict(X_test)

    # Assessment of classification accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generating a classification report
    report = classification_report(y_test, y_pred, zero_division=1)
    # train_and_save_model(X_train, y_train, report)

    # Assuming map_emotion_label is a function that maps numerical labels to emotion types
    final_emotions = [map_emotion_label(labels) for labels in y_pred]

    # Count occurrences of each emotion type
    emotion_counts = Counter(final_emotions)

    # Get the most common emotion type
    most_common_emotion = emotion_counts.most_common(1)[0][0]

    print("RF Most common emotion type:", most_common_emotion)

    print("Classification accuracy of 4 emotions : " + str(accuracy))
    print(report)

    # Show graph
    plt.show()
    plt.title('Random Forest Model Score', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(linestyle=':', axis='y')
    a = plt.bar("KNN", accuracy, 0.3, color='blue', label='test', align='center')
    plt.show()

    return accuracy

def classification_knn(eeg_band, labels, k=12):
    print("K Nearest neighbor classifications")
    """
    A function for training a classifier on a data set and performing classification.

    Arguments:
    - eeg_band: np.array, a data set of size (1280, 160) with EEG measurements.
    - labels: np.array, class labels of size (1280, 3) in the format [arousal (float), valence (float), emotion (string)].
    - k: int, number of neighbors for KNN classifier (default is 5).

    Returns:
    - accuracy: float, classification accuracy on test data.
    - report: str, classification report (precision, recall, f1-score) on test data.
    """

    labels_valence = []
    labels_arousal = []
    labels_emotion = []

    for la in labels:
        if la[0] > 0:
            labels_valence.append(1)
        else:
            labels_valence.append(0)
        if la[1] > 0:
            labels_arousal.append(1)
        else:
            labels_arousal.append(0)
        if la[2] == "happy":
            labels_emotion.append(0)
        elif la[2] == "angry":
            labels_emotion.append(1)
        elif la[2] == "sad":
            labels_emotion.append(2)
        elif la[2] == "calm":
            labels_emotion.append(3)

    X = eeg_band
    # Measuring Renewal
    poly = preprocessing.PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    X = preprocessing.normalize(X, norm='l1')

    emotion = labels[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, emotion, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Assuming map_emotion_label is a function that maps numerical labels to emotion types
    final_emotions = [map_emotion_label(labels) for labels in y_pred]

    # Count occurrences of each emotion type
    emotion_counts = Counter(final_emotions)

    # Get the most common emotion type
    most_common_emotion = emotion_counts.most_common(1)[0][0]

    # Count occurrences of each emotion type
    emotion_counts = Counter(final_emotions)

    # Print the counts of all emotion types
    print("Counts of all emotion types:", emotion_counts)

    print("Test RF:", final_emotions)

    print("RF Most common emotion type:", most_common_emotion)

    print("Accuracy:", accuracy)
    report = classification_report(y_test, y_pred, zero_division=1)
    print(report)

    # Show graph
    plt.show()
    plt.title('KNN Model Score', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(linestyle=':', axis='y')
    a = plt.bar("KNN", accuracy, 0.3, color='black', label='test', align='center')
    plt.show()

    return accuracy


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
def bandpower(data, sf, band):
    sns.set(font_scale=1.2)

    band = np.asarray(band)
    # Determine the lower and upper limits of the delta
    low, high = band
    #  Signal graph of the first trial and last channels
    # plt.plot(data, lw=1.5, color='k')
    # plt.xlabel('Time')
    # plt.ylabel('Voltage')
    # sns.despine()
    # ----------------------------------------------------
    # Determining window length (4 seconds)
    window = (2 / low) * sf
    frequency, psd = signal.welch(data, sf, nperseg=window)

    # Power spectrum graph
    # sns.set(font_scale=1.2, style='white')
    # plt.figure(figsize=(8, 4))
    # plt.plot(frequency, psd, color='k', lw=2)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    # plt.xlim([0, frequency.max()])
    # sns.despine()
    # plt.show()
    # -------------------------
    # Frequency resolution
    freq_res = frequency[1] - frequency[0]
    # Intersecting values in the frequency vector
    idx_band = np.logical_and(frequency >= low, frequency <= high)

    # Before calculating the average delta band power, we need to find the frequency ranges that cross the delta frequency band.
    # Plot the power spectral density and fill the theta region
    # plt.figure(figsize=(8, 4))
    # plt.plot(frequency, psd, lw=2, color='k')
    # plt.fill_between(frequency, psd, where=idx_band, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 50])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    # sns.despine()
    # ------------------------------------------------------

    band_power = simps(psd[idx_band], dx=freq_res)
    return band_power


def get_band_power(people, channel, band, eeg_data):
    bd = (0, 0)
    if band == "theta":
        bd = (4,7)
    elif band == "alpha":
        bd = (8, 13)
    elif band == "beta":
        bd = (14, 30)
    elif band == "gamma":
        bd = (31, 50)
    return bandpower(eeg_data[people, channel], 128., bd)



def accuracy_compare_plot(result_rf, result_svm, result_knn):
    accuracy = [result_rf, result_svm, result_knn]

    model_name = ["RF", "SVM", "KNN"]

    plt.title('Accuracy of Classification Model Scores', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(linestyle=':', axis='y')
    x = np.arange(3)
    color = [ 'blue', 'red', 'black']

    a = plt.bar(x, accuracy, 0.3, color=color, label='test', align='center')

    for i in a:
        h = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, h, '%.3f' % h, ha='center', va='bottom')
    plt.xticks(x, model_name, rotation=75)
    plt.legend(loc='lower right')
    plt.show()

# files = []
# for n in range(1, 33):
#     s = ''
#     if n < 10:
#         s += '0'
#     s += str(n)
#     files.append(s)
# print(files)
#
# print()

def train_and_save_KNN_model(model):

    # Save the trained model
    joblib.dump(model, '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/KNN_Model.pkl')

def train_and_save_SVM_model(model):

    # Save the trained model
    joblib.dump(model, '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/SVM_Model.pkl')

def train_and_save_RF_model(model):

    # Save the trained model
    joblib.dump(model, '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/RF_Model.pkl')

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

if __name__ == '__main__':
    # 1. Data Preprocessing
    # for i in files:
    raw_data_file = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/data_original/s28.bdf"
    raw_data = preprocessing_data(raw_data_file)

    if raw_data is not None:
        # 2. Allocation of initial dataframes for eeg data and features
        directory = "/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/data_preprocessed_matlab_test/"
        labels_for_feature, data_for_feature = read_preprocessed_data(directory)

        if labels_for_feature is not None and data_for_feature is not None:
            # 3. Feature extraction from eeg data based on the power range of each signal
            eeg_band_data = feature_extraction(labels_for_feature, data_for_feature)

            # 4 Labeling the arousal, valence training set with appropriate emotions
            # 1 - happy, 2 - angry, 3 - calm, 4 - sad.
            labels_for_classification = create_labels(labels_for_feature)

            left = np.array(["Fp1", "AF3", "F7", "FC5", "T7"])
            right = np.array(["Fp2", "AF4", "F8", "FC6", "T8"])
            frontal = np.array(["F3", "FC1", "Fz", "F4", "FC2"])
            parietal = np.array(["P3", "P7", "Pz", "P4", "P8"])
            occipital = np.array(["O1", "Oz", "O2", "PO3", "PO4"])
            central = np.array(["CP5", "CP1", "Cz", "C4", "C3", "CP6", "CP2"])

            # 5 Classification of different machine learning methods
            # data is taken from previously saved results of previous methods to speed up work
            labels_for_classification = '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Python/eeg_labels.npy'
            eeg_band_data = '/Users/chirathsirimanna/PycharmProjects/finalYearMlCode/Deap/Deap dataset/New_PreProcessed_Python/eeg_band.npy'


            # # classification with random_forest classification
            result_rf = random_forest_classifier(np.load(eeg_band_data, allow_pickle=True),
                np.load(labels_for_classification, allow_pickle=True))

            train_and_save_RF_model(result_rf)



            # # classification with svm classification
            result_svm = svm_classifier(np.load(eeg_band_data, allow_pickle=True),
                np.load(labels_for_classification, allow_pickle=True))

            train_and_save_SVM_model(result_svm)


            # # classification with knn classification
            result_knn = classification_knn(np.load(eeg_band_data, allow_pickle=True),
                np.load(labels_for_classification, allow_pickle=True))

            train_and_save_KNN_model(result_knn)

            # 6 FINAL ACCURACY RESULT
            accuracy_compare_plot(result_rf, result_svm, result_knn)