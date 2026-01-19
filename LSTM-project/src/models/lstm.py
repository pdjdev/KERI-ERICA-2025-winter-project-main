import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr  # 피어슨 상관계수 계산
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.loss")
warnings.filterwarnings("ignore", category=FutureWarning)