import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

sns.set_style('darkgrid')
sns.set(font_scale = 1.1)

    
def make_initial_imports():
    
    global pd, np, plt, sns, ticker
    return pd, np, plt, sns, ticker
    

def load_data(url):
    
    global df
    
    df = pd.read_csv(url, sep=';')
    
    return df