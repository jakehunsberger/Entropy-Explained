import numpy as np


def Entropy_Explained(Y_True, Y_Pred, Obs_Weights=None):
    
    #Parameter to undiscretize Y_True/Y_Pred
    Epsilon = 1e-15
    
    #Convert to numpy arrays if list
    Y_True = np.array(Y_True, dtype='float64')
    Y_Pred = np.array(Y_Pred, dtype='float64')
    
    #Default equal observation weights
    if Obs_Weights is None:
        Obs_Weights = np.ones(shape=(len(Y_True)))
        pass
    #User-defined observation weights
    elif Obs_Weights is not None:
        Obs_Weights = np.array(Obs_Weights, dtype='float64')
        pass
    
    #Check that data shapes across Y_True, Y_Pred & Obs_Weights are equal to each other
    Data_Shape = Y_True.shape
    if Y_Pred.shape != Data_Shape:
        print('Classification Metrics, Entropy Explained: Y_Pred shape not equal to that of Y_True')
        return None
    if Obs_Weights.shape != Data_Shape:
        print('Classification Metrics, Entropy Explained: Obs_Weights shape not equal to that of Y_True')
        return None
    
    #Reshape
    Y_True = Y_True.reshape(-1)
    Y_Pred = Y_Pred.reshape(-1)
    Obs_Weights = Obs_Weights.reshape(-1)
    
    #Filter out null valued observations
    Y_True_NonNulls_Bool = ~np.isnan(Y_True)
    Y_Pred_NonNulls_Bool = ~np.isnan(Y_Pred)
    Weights_NonNulls_Bool = ~np.isnan(Obs_Weights)
    Obs_NonNulls_Bool = Y_True_NonNulls_Bool * Y_Pred_NonNulls_Bool * Weights_NonNulls_Bool
    
    #Case where all observations contain a null value
    if np.sum(Obs_NonNulls_Bool) < 2:
        print('Classification Metrics, Entropy Explained: too many null values encountered across Y_True, Y_Pred or Obs_Weights')
        return None
    
    #Omit any observations that contain a null value
    Y_True = Y_True[Obs_NonNulls_Bool]
    Y_Pred = Y_Pred[Obs_NonNulls_Bool]
    Obs_Weights = Obs_Weights[Obs_NonNulls_Bool]    
    
    #Check to ensure variation in true response values
    if np.std(Y_True) == 0:
        print('Classification Metrics, Entropy Explained:\n')
        print('Y_True presents no variation of class probabilities')
        return None
    
    #Normalize the observation weights
    Obs_Weights = Obs_Weights / np.sum(Obs_Weights)
    
    #Baseline Probability Measure
    Baseline_Probability = np.sum(Obs_Weights * Y_True)
    
    #Clip both Y_True/Y_Pred to epsilon
    Y_True[Y_True < Epsilon] = Epsilon
    Y_True[Y_True > (1 - Epsilon)] = 1 - Epsilon
    Y_Pred[Y_Pred < Epsilon] = Epsilon
    Y_Pred[Y_Pred > (1 - Epsilon)] = 1 - Epsilon
    
    #Your model's predictive capability
    Obs_Residual_CrossEntropy = -1 * (Y_True * np.log(Y_Pred) + (1-Y_True) * np.log(1-Y_Pred))
    
    #Worst possible model predictive capability
    Obs_Baseline_CrossEntropy = -1 * (Y_True * np.log(Baseline_Probability) + (1-Y_True) * np.log(1-Baseline_Probability))
    
    #Best possible model predictive capability
    Obs_Information_Entropy = -1 * (Y_True * np.log(Y_True) + (1-Y_True) * np.log(1-Y_True))
    
    #Kullback-Leibler divergences for model and baseline
    Obs_KL_Div_Residual = Obs_Residual_CrossEntropy - Obs_Information_Entropy
    Obs_KL_Div_Baseline = Obs_Baseline_CrossEntropy - Obs_Information_Entropy
    
    #Sum up individual weighted entropic divergence measures
    KL_Div_Residual = np.sum(Obs_Weights * Obs_KL_Div_Residual)
    KL_Div_Baseline = np.sum(Obs_Weights * Obs_KL_Div_Baseline)
    
    #Final expression for entropy explained
    Entropy_Explained = 1 - abs(KL_Div_Residual) / KL_Div_Baseline
    return Entropy_Explained








