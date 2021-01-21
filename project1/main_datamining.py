from numpy import zeros
from sklearn import linear_model,svm
import pandas as pd

import fitting_scoring
import process_input

# ------------------------------------------------------------------------------------------------
descriptors_file = "Practice_Descriptors.csv"
targets_file = "Practice_Targets.csv"

# ------------------------------------------------------------------------------------------------
# Step 1
descriptors = process_input.open_descriptor_matrix(descriptors_file)

targets = process_input.open_target_values(targets_file)
print("\nShape of descriptors before cleaning :", descriptors.shape)
# ------------------------------------------------------------------------------------------------
# Step 2
# Filter out molecules with NaN-value descriptors and descriptors with little or no variance
descriptors, targets = process_input.removeInvalidData(descriptors, targets)
print("\nShape of descriptors after cleaning :", descriptors.shape)
descriptors, active_descriptors = process_input.removeNearConstantColumns(descriptors.to_numpy())
# Rescale the descriptor data
descriptors = process_input.rescale_data(pd.DataFrame(descriptors))

# ------------------------------------------------------------------------------------------------
# Step 3
descriptors, targets = process_input.sort_descriptor_matrix(descriptors.to_numpy(), targets.to_numpy())

# ------------------------------------------------------------------------------------------------
# Step 4
X_Train, X_Valid, X_Test, Y_Train, Y_Valid, Y_Test = process_input.simple_split(descriptors, targets)
data = {'TrainX': X_Train, 'TrainY': Y_Train, 'ValidateX': X_Valid, 'ValidateY': Y_Valid,
        'TestX': X_Test, 'TestY': Y_Test, 'UsedDesc': active_descriptors}

print(str(descriptors.shape[1]) + " valid descriptors and " + str(targets.__len__()) + " molecules available.")

#print(X_Train[0:5, 0:20])

# ------------------------------------------------------------------------------------------------
# Step 5
# Set up the demonstration model
featured_descriptors = [4, 8, 12, 16]  # These indices are "false", applying only to the truncated post-filter descriptor matrix.
binary_model = zeros((1, X_Train.shape[1]))

binary_model[0][featured_descriptors] = 1


# ------------------------------------------------------------------------------------------------
# Step 6
# Create a Multiple Linear Regression object to fit our demonstration model to the data
#regressor = linear_model.LinearRegression()
regressor = svm.SVR()
instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'svm'}

trackDesc, trackFitness, trackModel, \
trackDimen, trackR2train, trackR2valid, \
trackR2test, testRMSE, testMAE, \
testAccPred = fitting_scoring.evaluate_population(model=regressor, instructions=instructions, data=data,population=binary_model, exportfile=None)

'''clf = svm.SVC(kernel='linear')  #linear classifier
instructions2 = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'svm'}
trackDesc2, trackFitness2, trackModel2, \
trackDimen2, trackR2train2, trackR2valid2, \
trackR2test2, testRMSE2, testMAE2, \
testAccPred2 = fitting_scoring.evaluate_population(model=clf, instructions=instructions2, data=data,
                                                       population=binary_model, exportfile=None)'''

# ------------------------------------------------------------------------------------------------
# Step 7
for key in trackDesc.keys():
    print("Descriptors:")
    print("\t" + str(trackDesc[key]))  # This will show the "true" indices of the featured descriptors in the full matrix
    print("Fitness:")
    print("\t" + str(trackFitness[key]))
    print("Model:")
    print("\t" + str(trackModel[key]))
    print("Dimensionality:")
    print("\t" + str(trackDimen[key]))
    print("R2_Train:")
    print("\t" + str(trackR2train[key]))
    print("R2_Valid:")
    print("\t" + str(trackR2valid[key]))
    print("R2_Test:")
    print("\t" + str(trackR2test[key]))
    print("Testing RMSE:")
    print("\t" + str(testRMSE[key]))
    print("Testing MAE:")
    print("\t" + str(testMAE[key]))
    print("Acceptable Predictions From Testing Set:")
    print("\t" + str(100*testAccPred[key]) + "% of predictions")
'''for key in trackDesc2.keys():
    print("Descriptors:")
    print("\t" + str(trackDesc2[key]))  # This will show the "true" indices of the featured descriptors in the full matrix
    print("Fitness:")
    print("\t" + str(trackFitness2[key]))
    print("Model:")
    print("\t" + str(trackModel2[key]))
    print("Dimensionality:")
    print("\t" + str(trackDimen2[key]))
    print("R2_Train:")
    print("\t" + str(trackR2train2[key]))
    print("R2_Valid:")
    print("\t" + str(trackR2valid2[key]))
    print("R2_Test:")
    print("\t" + str(trackR2test2[key]))
    print("Testing RMSE:")
    print("\t" + str(testRMSE2[key]))
    print("Testing MAE:")
    print("\t" + str(testMAE2[key]))
    print("Acceptable Predictions From Testing Set:")
    print("\t" + str(100*testAccPred2[key]) + "% of predictions")'''

# ------------------------------------------------------------------------------------------------
# Step 8
