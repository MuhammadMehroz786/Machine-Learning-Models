# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from python_speech_features import mfcc
import librosa
import os

#augment audio data
def augment_audio(audio):
    noise = np.random.randn(len(audio))
    audio_noise = audio + 0.005 * noise
    shift = np.random.randint(0, 1000)
    audio_shift = np.roll(audio, shift)
    return audio_noise, audio_shift

# extract features from audio files
def extract_features(audio_path, max_len=500):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, res_type='kaiser_fast')
        audio_noise, audio_shift = augment_audio(audio)
        mfccs = mfcc(audio, samplerate=sr, nfft=512, winlen=0.025, winstep=0.01, numcep=13, nfilt=26)
        mfccs_noise = mfcc(audio_noise, samplerate=sr, nfft=512, winlen=0.025, winstep=0.01, numcep=13, nfilt=26)
        mfccs_shift = mfcc(audio_shift, samplerate=sr, nfft=512, winlen=0.025, winstep=0.01, numcep=13, nfilt=26)
        
        def pad_or_truncate(mfccs):
            if mfccs.shape[0] < max_len:
                pad_width = max_len - mfccs.shape[0]
                return np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
            else:
                return mfccs[:max_len, :]
        
        mfccs = pad_or_truncate(mfccs)
        mfccs_noise = pad_or_truncate(mfccs_noise)
        mfccs_shift = pad_or_truncate(mfccs_shift)
        
        return [mfccs, mfccs_noise, mfccs_shift]
    except Exception as e:
        print(f"Error encountered while parsing audio file: {audio_path}. Error: {e}")
        return None

# Load dataset and preprocess
def load_and_preprocess_data(data_dir):
    emotions = {'angry': 0, 'happy': 1, 'neutral': 2, 'sad': 3}  # Add more emotions as needed
    X, y = [], []

    for emotion, label in emotions.items():
        emotion_folder = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_folder):
            audio_path = os.path.join(emotion_folder, filename)
            if filename.endswith('.wav'):
                features_list = extract_features(audio_path)
                if features_list is not None:
                    for features in features_list:
                        X.append(features)
                        y.append(label)

    X = np.array(X)
    y = to_categorical(y)

    return X, y

# Define the model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and preprocess data
data_directory = './Emotion Dataset'
X, y = load_and_preprocess_data(data_directory)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for the Conv2D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Build and train the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = build_model(input_shape, num_classes)

# Define a learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Test Accuracy: {:.2f}%".format(accuracy * 100))
