import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from common import logger
from common import containers_info

# Environment Variables
config = os.environ.get("DATASET") # Should point to a CSV or config file in /app/dataset
logging_level = logger.getLevelName(os.environ.get("LOGGING_LEVEL", "INFO"))
plot_perf = 0 if os.environ.get("PLOT_PERFORMANCE", "False").lower() == "false" else 1
instrument_profile = os.environ.get("INSTRUMENT_PROFILE", "drums") # drums, bass, vocals, other

# Configure logging
log = logger.setup_logger(containers_info.get_current_container_name(), logging_level)

occupied = 0
global_filename = str(config).split(".")[0]
current_run_path = ""

# Data holders
x_train, y_train_class, y_train_sep = [], [], []
x_test, y_test_class, y_test_sep = [], [], []
x_pred, y_pred_class, y_pred_sep = [], [], []

# Constants
IMG_SIZE = 128
INSTRUMENTS = ['drums', 'bass', 'vocals', 'other']
INST_TO_IDX = {inst: i for i, inst in enumerate(INSTRUMENTS)}

###########################################################################################
# AUDIO UTILS #############################################################################
###########################################################################################

def preprocess_audio(file_path, duration=5.0, sr=22050):
    try:
        y, _ = librosa.load(file_path, duration=duration, sr=sr)
        if len(y) < duration * sr:
            y = np.pad(y, (0, int(duration * sr) - len(y)))
        
        # Compute Mel Spectrogram
        spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE)
        spec_db = librosa.power_to_db(spec, ref=np.max)
        
        # Normalize to [0, 1]
        spec_db = (spec_db + 80) / 80
        
        # Resize/Crop to IMG_SIZE x IMG_SIZE
        if spec_db.shape[1] > IMG_SIZE:
            spec_db = spec_db[:, :IMG_SIZE]
        elif spec_db.shape[1] < IMG_SIZE:
            spec_db = np.pad(spec_db, ((0, 0), (0, IMG_SIZE - spec_db.shape[1])))
            
        return spec_db.reshape(IMG_SIZE, IMG_SIZE, 1)
    except Exception as e:
        log.error(f"Error processing {file_path}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE, 1))

###########################################################################################
# MODEL DESCRIPTION #######################################################################
###########################################################################################

def build_multi_task_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Shared Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    latent = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Perception Head (Multi-label Classification)
    gap = layers.GlobalAveragePooling2D()(latent)
    perc = layers.Dense(64, activation='relu')(gap)
    perception_output = layers.Dense(len(INSTRUMENTS), activation='sigmoid', name='perception')(perc)
    
    # Separation Heads
    def separation_head(name):
        d = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(latent)
        d = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(d)
        d = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same', name=name)(d)
        return d
    
    sep_heads = [separation_head(f"sep_{inst}") for inst in INSTRUMENTS]
    
    model = models.Model(inputs=inputs, outputs=[perception_output] + sep_heads)
    
    # Compile with appropriate losses
    # We use a custom training step or masking if needed, but for fit() we'll use:
    losses = {
        'perception': 'binary_crossentropy'
    }
    for inst in INSTRUMENTS:
        losses[f'sep_{inst}'] = 'mse'
        
    model.compile(optimizer='adam', loss=losses, metrics={'perception': 'accuracy'})
    return model

model = build_multi_task_model()

###########################################################################################
# FUNCTIONS FOR MODEL #####################################################################
###########################################################################################

def in_usage(value):
    global occupied
    occupied = value
    if value == 1:
        log.info("ML app is now in use, and blocked")
    elif value == 0:
        log.info("ML app is now freed")

def set_run_path(run_path):
    global current_run_path
    current_run_path = run_path
    log.info("Current run Path set")

def get_parameters():
    global model
    return model.get_weights()

def set_parameters(parameters):
    global model
    model.set_weights(parameters)
    log.info("Parameters Set!")

def get_data():
    global x_train, y_train_class, y_train_sep
    global x_test, y_test_class, y_test_sep
    global instrument_profile, config
    
    base_dir = "/app/dataset"
    
    # Check if config file exists and use it to filter songs
    selected_songs = None
    if config and os.path.exists(os.path.join(base_dir, config)):
        try:
            df = pd.read_csv(os.path.join(base_dir, config))
            if 'song_name' in df.columns:
                selected_songs = df['song_name'].tolist()
                log.info(f"Using config file {config} to select {len(selected_songs)} songs.")
        except Exception as e:
            log.error(f"Error reading config file {config}: {e}")

    # Find all songs in train and test
    songs = []
    for root, dirs, files in os.walk(base_dir):
        if "mixture.wav" in files:
            song_name = os.path.basename(root)
            if selected_songs is None or song_name in selected_songs:
                songs.append(root)
    
    log.info(f"Found {len(songs)} songs in {base_dir}")
    
    # Limit to a subset for prototype speed if no config is provided
    if selected_songs is None:
        songs = songs[:20] 
    
    X, Y_class, Y_sep = [], [], []
    
    for song_path in songs:
        # Load mixture
        mix_spec = preprocess_audio(os.path.join(song_path, "mixture.wav"))
        X.append(mix_spec)
        
        # Classification labels (all instruments present in MUSDB)
        # In a real scenario, we'd check if the file exists and has signal
        Y_class.append([1.0, 1.0, 1.0, 1.0]) 
        
        # Separation target for THIS node's profile
        target_path = os.path.join(song_path, f"{instrument_profile}.wav")
        if os.path.exists(target_path):
            target_spec = preprocess_audio(target_path)
        else:
            target_spec = np.zeros((IMG_SIZE, IMG_SIZE, 1))
        Y_sep.append(target_spec)
        
    X = np.array(X)
    Y_class = np.array(Y_class)
    Y_sep = np.array(Y_sep)
    
    # Split
    x_train, x_test, y_train_class, y_test_class, y_train_sep, y_test_sep = train_test_split(
        X, Y_class, Y_sep, test_size=0.2, random_state=42
    )
    
    log.info(f"Data Loaded: Train={len(x_train)}, Test={len(x_test)}")

def fit(config):
    log.info("----------------  FIT  ----------------- ")
    global model, x_train, y_train_class, y_train_sep, instrument_profile
    
    epochs = int(config.get("epochs", 1))
    batch_size = int(config.get("batch_size", 4))
    
    # Create target dict for multi-output
    # We only have ground truth for our profile's separation head.
    # For others, we'll pass zeros but we should ideally not train them (freeze or use custom loss)
    # Simple approach: pass our Y_sep to our head, and dummy to others.
    
    y_train_dict = {'perception': y_train_class}
    for inst in INSTRUMENTS:
        if inst == instrument_profile:
            y_train_dict[f'sep_{inst}'] = y_train_sep
        else:
            y_train_dict[f'sep_{inst}'] = np.zeros_like(y_train_sep)
            
    history = model.fit(
        x_train,
        y_train_dict,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    results = {
        "loss": history.history["loss"][-1],
        "perception_accuracy": history.history["perception_accuracy"][-1]
    }
    
    return len(x_train), results

def evaluate(config):
    log.info("----------------  EVALUATE  ----------------- ")
    global model, x_test, y_test_class, y_test_sep, instrument_profile
    
    y_test_dict = {'perception': y_test_class}
    for inst in INSTRUMENTS:
        if inst == instrument_profile:
            y_test_dict[f'sep_{inst}'] = y_test_sep
        else:
            y_test_dict[f'sep_{inst}'] = np.zeros_like(y_test_sep)
            
    eval_results = model.evaluate(x_test, y_test_dict, verbose=0)
    # eval_results order: [total_loss, perception_loss, sep_drums_loss, ..., perception_acc, ...]
    
    log.info(f"Model metrics names: {model.metrics_names}")
    loss = eval_results[0]
    
    # In newer Keras, the output name might be prepended
    if 'perception_accuracy' in model.metrics_names:
        acc_idx = model.metrics_names.index('perception_accuracy')
    elif 'perception_acc' in model.metrics_names:
        acc_idx = model.metrics_names.index('perception_acc')
    else:
        # Fallback to finding anything with 'accuracy'
        acc_idx = next((i for i, name in enumerate(model.metrics_names) if 'accuracy' in name), 1)
        
    accuracy = eval_results[acc_idx]
    
    return float(loss), len(x_test), {"perception_accuracy": float(accuracy)}

def predict(plot_graphs, before_after, centralized_predict="false"):
    log.info("----------------  PREDICT  ----------------- ")
    global model, x_test, instrument_profile
    
    # Select first sample from test set
    sample_mix = x_test[0:1]
    outputs = model.predict(sample_mix)
    
    # outputs: [perception, sep_drums, sep_bass, sep_vocals, sep_other]
    perc_probs = outputs[0][0]
    
    log.info(f"Perception Probabilities: {dict(zip(INSTRUMENTS, perc_probs))}")
    
    if plot_graphs != "false":
        plot_results(sample_mix[0], outputs, before_after)
        
    return {"loss": 0.0}, {"perception_accuracy": 0.0} # Dummies for interface compatibility

def plot_results(mix_spec, outputs, before_after):
    # Plot mixture and separated instruments
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(mix_spec.reshape(IMG_SIZE, IMG_SIZE), aspect='auto', origin='lower')
    plt.title("Mixture Spectrogram")
    
    for i, inst in enumerate(INSTRUMENTS):
        plt.subplot(2, 3, i+2)
        sep_spec = outputs[i+1][0].reshape(IMG_SIZE, IMG_SIZE)
        plt.imshow(sep_spec, aspect='auto', origin='lower')
        plt.title(f"Separated: {inst}")
        
    plt.tight_layout()
    plt.savefig(f"{current_run_path}/separation_{before_after}.png")
    plt.close()
