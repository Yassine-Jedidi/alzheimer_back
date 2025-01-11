import pandas as pd

# for pre processing
import librosa
import numpy as np
import soundfile as sf
from keras.models import load_model
from scipy.signal import butter, filtfilt
import joblib
from scipy.interpolate import interp1d
import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for binary classification.
    Args:
        gamma (float): Focusing parameter. Default is 2.0.
        alpha (float): Balancing factor for positive/negative classes. Default is 0.25.
    Returns:
        A callable loss function to use with Keras.
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute focal loss components
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 -
                      y_pred)  # Probability of true class
        ce = -K.log(pt)  # Cross-entropy
        fl = alpha * K.pow(1 - pt, gamma) * ce  # Focal loss formula

        return K.mean(fl)
    return loss


def preprocess_audio(input_path, output_path="D:\\alzheimer\\new_audio.wav"):
    """
    Enhanced preprocessing pipeline for better diarization:
    1. Load and convert to mono 16kHz (required by pyannote)
    2. Remove DC offset
    3. Apply pre-emphasis filter
    4. Apply bandpass filter focusing on speech frequencies (100-8000 Hz)
    5. Remove background noise using spectral gating
    6. Apply volume normalization
    7. Trim leading and trailing silence
    """
    print(f"\nPreprocessing {input_path}...")

    # Load audio file and resample to 16kHz (required by pyannote)
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    print(f"Loaded audio: {y.shape} samples, {sr}Hz")

    # Remove DC offset
    y = librosa.util.normalize(y - np.mean(y))
    print("Removed DC offset")

    # Apply pre-emphasis filter to enhance high frequencies
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    print("Applied pre-emphasis filter")

    # Bandpass filter
    nyquist = sr / 2
    low_freq = 100  # Hz
    high_freq = min(8000, nyquist-1)  # Hz
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(4, [low, high], btype='band')
    y = filtfilt(b, a, y)
    print(f"Applied bandpass filter ({low_freq}-{high_freq} Hz)")

    # Enhanced noise reduction using spectral gating
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    mag, phase = librosa.magphase(D)
    mag_db = librosa.amplitude_to_db(mag)
    noise_floor = np.percentile(mag_db, 20, axis=1)[:, np.newaxis]
    threshold_db = noise_floor + 10
    mask = librosa.db_to_amplitude(mag_db - threshold_db)
    mask = np.maximum(0, np.minimum(1, mask))
    mag_cleaned = mag * mask
    y = librosa.istft(mag_cleaned * phase, hop_length=512)
    print("Applied enhanced noise reduction")

    # Apply dynamic range compression
    y = np.sign(y) * np.log1p(np.abs(y) * 10) / np.log(10)

    # Final normalization
    y = librosa.util.normalize(y, norm=np.inf)
    print("Applied final normalization")

    # Trim leading and trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)
    print("Trimmed leading and trailing silence")

    # Save preprocessed audio
    sf.write(output_path, y, sr)
    print(f"Saved preprocessed audio to {output_path}")

    return output_path


def predict_alzheimer(audio_path, model_path='alzheimer_model.h5',
                      scaler_path='alzheimer_scaler.pkl',
                      labels_path='alzheimer_labels.npy'):
    """
    Make prediction on a new audio file using the saved model
    """
    print("Starting prediction process...")

    # Load the saved model and preprocessing objects
    print("Loading model and preprocessing objects...")
    custom_objects = {
        'loss': focal_loss(gamma=2., alpha=0.25),
        'focal_loss': focal_loss(gamma=2., alpha=0.25)
    }

    model = tf.keras.models.load_model(
        model_path, custom_objects=custom_objects)
    scaler = joblib.load(scaler_path)
    labels = np.load(labels_path, allow_pickle=True)

    print("Model and preprocessing objects loaded successfully")
    print(f"Available labels: {labels}")

    # No need to preprocess audio again, as it has already been done
    processed_audio_path = audio_path  # The already preprocessed audio file

    # Extract MFCC features
    print("Extracting MFCC features...")
    y, sr = librosa.load(processed_audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    print(f"MFCC shape before interpolation: {mfccs.shape}")

    # Flatten MFCCs before interpolation into a single row
    flattened_mfccs_before = mfccs.flatten()
    column_names_before = [
        f"mfcc_{frame + 1}_{coeff + 1}"
        for frame in range(mfccs.shape[1])
        for coeff in range(mfccs.shape[0])
    ]
    df_before_interpolation = pd.DataFrame(
        [flattened_mfccs_before], columns=column_names_before)

    print("\nMFCCs Before Interpolation (Flattened DataFrame):")
    print(df_before_interpolation.head())

    # Apply linear interpolation to match target frames
    target_frames = 909  # Replace with your model's expected number of frames
    current_frames = mfccs.shape[1]

    # Reshape MFCCs into a DataFrame for interpolation
    mfcc_df = pd.DataFrame(mfccs.T)  # Transpose for frame-major structure
    interpolated_mfcc_df = mfcc_df.reindex(
        range(target_frames))  # Add target frame indices
    interpolated_mfcc_df.interpolate(
        method='linear', axis=0, inplace=True)  # Interpolate missing values
    # Convert back and transpose to original shape
    interpolated_mfccs = interpolated_mfcc_df.to_numpy().T

    print(f"MFCC shape after interpolation: {interpolated_mfccs.shape}")

    # Flatten MFCCs after interpolation into a single row
    flattened_mfccs_after = interpolated_mfccs.flatten()
    column_names_after = [
        f"mfcc_{frame + 1}_{coeff + 1}"
        for frame in range(interpolated_mfccs.shape[1])
        for coeff in range(interpolated_mfccs.shape[0])
    ]
    df_after_interpolation = pd.DataFrame(
        [flattened_mfccs_after], columns=column_names_after)

    print("\nMFCCs After Interpolation (Flattened DataFrame):")
    print(df_after_interpolation.head())

    # Reshape and scale features
    mfccs_scaled = scaler.transform([flattened_mfccs_after])
    X = mfccs_scaled.reshape(1, target_frames, 13)
    print(f"Final input shape: {X.shape}")

    # Make prediction
    print("Making prediction...")
    prediction = model.predict(X)

    # Handle DeprecationWarning by extracting the scalar value correctly
    if prediction.ndim == 2 and prediction.shape[1] > 1:
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]
    else:
        predicted_class = int(np.round(prediction[0][0]))
        confidence = prediction[0][0]

    predicted_label = labels[predicted_class]
    print(f"Predicted class: {predicted_class} ({predicted_label})")

    if predicted_label == "control":
        confidence = 1 - confidence

    return df_before_interpolation, df_after_interpolation, predicted_label, confidence
