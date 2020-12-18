import numpy as np
from Entropy_Explained import Entropy_Explained

#Test cases

#Accurate predictions
Y_True = np.array([0,1])
Y_Pred = np.array([0,1])
R2 = Entropy_Explained(Y_True, Y_Pred)
print('Completely Accurate Model R2: ' + str(R2))

#Completely inaccurate predictions
Y_True = np.array([0,1])
Y_Pred = np.array([1,0])
R2 = Entropy_Explained(Y_True, Y_Pred)
print('Completely Inaccurate Model R2: ' + str(R2))


#Under confidence (Normal)
#Yields strong positive value
Y_True = np.array([0.25,0.75])
Y_Pred = np.array([0.3,0.7])
R2 = Entropy_Explained(Y_True, Y_Pred)
print('Under Confident Model R2: ' + str(R2))

#Over confidence (Abnormal)
Y_True = np.array([0.25,0.75])
Y_Pred = np.array([0.1,0.9])
R2 = Entropy_Explained(Y_True, Y_Pred)
print('Over Confident Model R2: ' + str(R2))


'''
Obs_Weights = np.array([0.9,0.1])
R2 = Entropy_Explained(Y_True, Y_Pred, Obs_Weights)
print(R2)
'''






