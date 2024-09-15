import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open('crop_recommendation_model .pkl', 'rb'))
