# Core deep learning libraries (PyTorch for Python 3.13 compatibility)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import vgg16, VGG16_Weights

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
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)