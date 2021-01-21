from numpy import zeros
from sklearn import linear_model,svm
import pandas as pd

import fitting_scoring
import process_input
class DrugDiscovery:
        def __init__(self, descriptors_file,targets_file):
            descriptors_file = descriptors_file
            targets_file = targets_file
            self.descriptors = process_input.open_descriptor_matrix(descriptors_file)
            self.targets = process_input.open_target_values(targets_file)
        def removeInvalidData(self):
            self.descriptors, self.targets = process_input.removeInvalidData(self.descriptors, self.targets)

        def removeNearConstantColumns(self):
            self.descriptors, self.active_descriptors = process_input.removeNearConstantColumns(self.descriptors.to_numpy())

        def rescale_data(self):
            self.descriptors = process_input.rescale_data(pd.DataFrame(self.descriptors))

        def sort_descriptor_matrix(self):
            self.descriptors, self.targets = process_input.sort_descriptor_matrix(self.descriptors.to_numpy(), self.targets.to_numpy())
        def simple_split(self):
            self.X_Train,self.X_Valid,self.X_Test, self.Y_Train, self.Y_Valid, self.Y_Test = process_input.simple_split(self.descriptors, self.targets)
            self.data = {'TrainX':self.X_Train, 'TrainY': self.Y_Train, 'ValidateX': self.X_Valid, 'ValidateY': self.Y_Valid,
                    'TestX': self.X_Test, 'TestY': self.Y_Test, 'UsedDesc': self.active_descriptors}
        def binaryModel(self):
            self.featured_descriptors = [4, 8, 12, 16]  # These indices are "false", applying only to the truncated post-filter descriptor matrix.
            self.binary_model = zeros((1, self.X_Train.shape[1]))
            self.binary_model[0][self.featured_descriptors] = 1

        def modeling(self, instruction):
            self.regressor = linear_model.LinearRegression()
            self.instructions = instructions

            self.trackDesc, self.trackFitness, self.trackModel, \
            self.trackDimen, self.trackR2train, self.trackR2valid, \
            self.trackR2test, self.testRMSE, self.testMAE, \
            self.testAccPred = fitting_scoring.evaluate_population(model=self.regressor, instructions=self.instructions, data=self.data,population=self.binary_model, exportfile=None)

# ------------------------------------------------------------------------------------------------
descriptors_file = "Practice_Descriptors.csv"
targets_file = "Practice_Targets.csv"
Alzheimer = DrugDiscovery(descriptors_file,targets_file)
# ------------------------------------------------------------------------------------------------
# Step 1

print("\ndescriptor's shape:\n",Alzheimer.descriptors.shape)
# ------------------------------------------------------------------------------------------------
# Step 2
# Filter out molecules with NaN-value descriptors and descriptors with little or no variance
Alzheimer.removeInvalidData()
print("\ndescriptor's shape after cleaning:\n",Alzheimer.descriptors.shape)


Alzheimer.removeNearConstantColumns()
# Rescale the descriptor data
Alzheimer.rescale_data()

# ------------------------------------------------------------------------------------------------
# Step 3
Alzheimer.sort_descriptor_matrix()

# ------------------------------------------------------------------------------------------------
# Step 4
Alzheimer.simple_split()


print(str(Alzheimer.descriptors.shape[1]) + " valid descriptors and " + str(Alzheimer.targets.__len__()) + " molecules available.")

#print(X_Train[0:5, 0:20])

# ------------------------------------------------------------------------------------------------
# Step 5
# Set up the demonstration model
Alzheimer.binaryModel()
# ------------------------------------------------------------------------------------------------
# Step 6
# Create a Multiple Linear Regression object to fit our demonstration model to the data


instructions = {'dim_limit': 4, 'algorithm': 'None', 'MLM_type': 'MLR'}
Alzheimer.modeling(instructions)



# ------------------------------------------------------------------------------------------------
# Step 7
for key in Alzheimer.trackDesc.keys():
    print("Descriptors:")
    print("\t" + str(Alzheimer.trackDesc[key]))  # This will show the "true" indices of the featured descriptors in the full matrix
    print("Fitness:")
    print("\t" + str(Alzheimer.trackFitness[key]))
    print("Model:")
    print("\t" + str(Alzheimer.trackModel[key]))
    print("Dimensionality:")
    print("\t" + str(Alzheimer.trackDimen[key]))
    print("R2_Train:")
    print("\t" + str(Alzheimer.trackR2train[key]))
    print("R2_Valid:")
    print("\t" + str(Alzheimer.trackR2valid[key]))
    print("R2_Test:")
    print("\t" + str(Alzheimer.trackR2test[key]))
    print("Testing RMSE:")
    print("\t" + str(Alzheimer.testRMSE[key]))
    print("Testing MAE:")
    print("\t" + str(Alzheimer.testMAE[key]))
    print("Acceptable Predictions From Testing Set:")
    print("\t" + str(100*Alzheimer.testAccPred[key]) + "% of predictions")

# ------------------------------------------------------------------------------------------------
# Step 8
