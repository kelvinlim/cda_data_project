import numpy as np
import statsmodels.api as sm

size = 100
# Simulate data
Pill = np.random.normal(0, 1, size)
Y = np.random.normal(0, 1, size)
Out = -0.8 * Y + np.random.normal(0, 1, size)
X = 0.7 * Pill + 0.83 * Y + np.random.normal(0, 1, size)
C = 0.94 * X + np.random.normal(0, 1, size)

# write out data to a csv file
np.savetxt('data.csv', np.column_stack((Out, Pill, X, Y, C)), delimiter=',', header='Out,Pill,X,Y,C', comments='')
# Model 1: Out ~ Pill
X1 = sm.add_constant(Pill)  # Add a constant (intercept)
model1 = sm.OLS(Out, X1).fit()
print(model1.summary())

# Model 2: Out ~ Pill + X
X2 = sm.add_constant(np.column_stack((Pill, X)))
model2 = sm.OLS(Out, X2).fit()
print(model2.summary())

# Model 3: Out ~ Pill + C
X3 = sm.add_constant(np.column_stack((Pill, C)))
model3 = sm.OLS(Out, X3).fit()
print(model3.summary())