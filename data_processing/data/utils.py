import numpy as np
import pandas as pd
from scipy import stats

def find_outliers(col, threshold=3):
    z = np.abs(stats.zscore(col))
    idx_outliers = np.where(z>threshold, True, False)
    return pd.Series(idx_outliers, index=col.index)
 