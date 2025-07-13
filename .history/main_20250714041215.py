# Core deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, Dropout, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Video and image processing
import cv2
import mediapipe as mp
from PIL import Image

# Data manipulation and analysis
import numpy as np
import pandas as pd

# File and system operations
import os
import json
import pickle
from pathlib import Path

# Data splitting and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Visualization (optional but useful)
import matplotlib.pyplot as plt
import seaborn as sns

# Audio processing (if combining with audio)
import librosa
import soundfile as sf

# Utilities
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)