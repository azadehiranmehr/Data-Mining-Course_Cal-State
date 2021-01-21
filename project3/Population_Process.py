
from sklearn import linear_model , svm,svm,neural_network
import pandas as pd
import numpy as np
import random
import fitting_scoring
import math
import os
class Differential_Evolution:
       def __init__(self, model,data):
            #class constructor to initiat a Differential_Evolution class and initiate the variables
            #based on descriptors and targets file and the model(MLR, SCM or ANN)  that user seleted in the main program
            self.model = model
            if self.model == 0:
                print("you selected:", 'MLR model')
                self.regressor = linear_model.LinearRegression()
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'MLR'}
                self.filename = "mlr.csv"
            elif self.model == 1:
                print("you selected:", 'svm model')
                self.regressor = svm.SVR()
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'SVM'}
                self.filename ="svm.csv"
            elif self.model == 2:
                print("you selected:", 'ANN model')
                self.regressor = neural_network.MLPRegressor( max_iter=600)
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'ANN'}
                self.filename ="ann.csv"
            #copy data that what to run GA for them. data was store in dictionarydata type in previous steps(in process_input module)
            self.data = data
            self.TrainX = self.data['TrainX']
            self.popNum = 50
            self.best_fitness = 100000
            self.F = 0.7
            self.CV = 0.7

# ------------------------------------------------------------------------------------------------------------------------------------------
       def initial_population_total(self):
            # Set up an initial array of all zeros
            self.population = np.zeros((self.popNum, self.TrainX.shape[1]))
            for i in range(self.popNum):
                #produce total feature size float random number between 0 and 1
                index = np.random.rand(self.TrainX.shape[1])
                # assigne 1 to each index that has random number <= 0.015
                self.population[i, index <= 0.015] = 1
                # num_features: sum of indexed contain one
                num_feature = np.sum(self.population[i])
            #make sure that number of selected feature is between 5 and 25
                while (num_feature < 5 or num_feature > 25):
                    index = np.random.rand(self.TrainX.shape[1])
                    self.population[i, index < 0.015] = 1
                    num_feature = np.sum(self.population[i])
#------------------------------------------------------------------------------------------------------------------------------------------
       def modeling(self):
            #create fitting object(self.fit) from module fitting_scoring
            self.fit = fitting_scoring.fitting()
            #do evaluation process based on regressr(the type of model that was selected) and input cleaned rescaled and splitted data
            self.trackDesc, self.trackFitness, self.trackModel, \
            self.trackDimen, self.trackR2train, self.trackR2valid, \
            self.trackR2test, self.testRMSE, self.testMAE, \
            self.testAccPred = self.fit.evaluate_population(model= self.regressor, instructions=self.instructions, data=self.data,population=self.population, exportfile='')
#------------------------------------------------------------------------------------------------------------------------------------------
       def fitness_sort(self):
            #sort Dataframe based on fitness
            self.df= self.df.sort_values('Fitness')
#------------------------------------------------------------------------------------------------------------------------------------------
       def selected_2_total(self, selected_arr):
            # convert selected row to row of population
            total_arr = np.zeros((1, self.TrainX.shape[1]))
            # assigne one to selected_arr
            total_arr[0, selected_arr] = 1
            return total_arr
#-------------------------------------------------------------------------------------------------------------------------------------
       def create_next_population(self):
            self.old_population = self.population
            #sort dataframe based on fitness
            self.fitness_sort()
            self.df1col = self.df.iloc[:, 0].apply(lambda x: eval(x.replace('_', ',')))
            self.selected_2_total(self.df1col.iloc[0])
            self.population = np.zeros((self.popNum, self.TrainX.shape[1]))
            self.population[0] = self.selected_2_total(self.df1col.iloc[0])
            for i  in range(1, self.popNum):
                # num_features: sum of indexed contain one
                num_feature = 0
                # make sure that number of selected feature is between 5 and 25
                while (num_feature < 5 or num_feature > 25):
                    v = np.zeros((1, self.TrainX.shape[1]))
                    a, b, c = random.sample(range(1, self.popNum), 3)
                    for j in range(self.TrainX.shape[1]):
                        v[0,j] = math.floor(abs (self.old_population[a, j] + (self.F * (self.old_population[b, j]- self.old_population[c, j])) ) )
                    for k in range( self.TrainX.shape[1]):
                        r = np.random.rand(1)
                        if (r < self.CV):
                             self.population[i, k] = v[0,k]
                        else:
                             self.population[i,k] = self.old_population[i,k]
                    num_feature = np.sum(self.population[i])


            #print("equality of pop1 and pop2 ", np.array_equal(self.population, self.old_population))
            #print("shape of new pop and old pop ", self.population.shape , self.old_population.shape)
# -------------------------------------------------------------------------------------------------------------------------------------
       def PrintModelResults(self, j):
            #create dataframe based on the dictionary of data that returned from fitting scoring module
            mydicts =[self.trackDesc,self.trackFitness, self.trackModel, self.trackDimen ,self.trackR2train, self.trackR2test, self.trackR2valid, self.testRMSE, self.testMAE, self.testAccPred]
            df = pd.concat([pd.Series(d) for d in mydicts], axis=1).fillna(0).T
            df.index = [ 'Descriptors','Fitness','Model','Dimen','R2train','R2test','R2Validation','RMSE','testMAE','testAccPred']
            self.df= df.T
            self.df = self.df.reset_index(drop=True)
            #assine best fitness of new population to new fitness and compare it to the best fitness of previous population
            self.fitness_sort()
            new_fitness = self.df.iloc[0, 1]
            if (new_fitness < self.best_fitness):
                self.best_fitness = new_fitness
                self.best_fitness_popNo = j
                print(" best fitness is {} for population number {}".format(self.best_fitness, self.best_fitness_popNo))
            self.save_to_file()

# -------------------------------------------------------------------------------------------------------------------------------------
       def save_to_file(self,):
            if os.path.isfile(self.filename):
                # Old data
                #oldFrame = pd.read_csv(self.filename)
                # Concat
                #df_diff = pd.concat([oldFrame, self.df], ignore_index=True).drop_duplicates()
                # Write new rows to csv file
                self.df.to_csv(self.filename, mode='a', header=False, index= False)
            else:  # else it doesn't exist to  append
                self.df.to_csv(self.filename, index= False)

# -------------------------------------------------------------------------------------------------------------------------------------
       def remove_redundunt_from_file(self,):
            dFrame = pd.read_csv(self.filename)
            dFrame = dFrame.drop_duplicates()
            dFrame.to_csv(self.filename)




