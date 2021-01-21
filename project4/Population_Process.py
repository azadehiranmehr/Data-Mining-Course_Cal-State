
from sklearn import linear_model , svm,svm,neural_network
import pandas as pd
import numpy as np
import random
import fitting_scoring
import math
import os
class PBSO_Algorithm:
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
                self.regressor = neural_network.MLPRegressor(max_iter=600)
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'ANN'}
                self.filename ="ann.csv"
            #copy data that what to run GA for them. data was store in dictionarydata type in previous steps(in process_input module)
            self.data = data
            self.TrainX = self.data['TrainX']
            self.popNum = 50
            self.best_fitness = 100000

# ------------------------------------------------------------------------------------------------------------------------------------------
       def createInitialVelocity(self):

           #How to find the initial velocity:
           velocity = np.zeros((self.popNum, self.TrainX.shape[1]))
           for i in range(self.popNum):
               velocity[i] = np.random.rand(self.TrainX.shape[1])
           return velocity
          # print("self.velocity[i,j]:\n", self.velocity)

# ------------------------------------------------------------------------------------------------------------------------------------------
       def updateVelocity(self, velocity, localBestMatrix, globalBestRow, population):
           #Find Velocity matrix as follow: (C1=2 ,C2=2 by default, inseriaWeight= 0.9)
           c1 = c2 = 2
           inseriaWeight = 0.9
           for i in range(self.popNum):
               for j in range(self.TrainX.shape[1]):
                   term1 = c1*random.random()*(localBestMatrix[i,j]- population[i,j])
                   term2 = c2*random.random()*(globalBestRow[j] - population[i,j])

                   velocity[i,j] = (inseriaWeight * velocity[i, j]) + term1 + term2
                   np.random.shuffle(velocity)
           return velocity



                   # ------------------------------------------------------------------------------------------------------------------------------------------
       def initialPopulation(self):
            # Set up an initial array of all zeros
            population = np.zeros((self.popNum, self.TrainX.shape[1]), dtype=float)
            for i in range(self.popNum):
                #produce total feature size float random number between 0 and 1
                index = np.random.rand(self.TrainX.shape[1])
                # assigne 1 to each index that has random number <= 0.015
                population[i, index <= 0.015] = 1
                # num_features: sum of indexed contain one
                num_feature = np.sum(population[i])
            #make sure that number of selected feature is between 5 and 25
                while (num_feature < 5 or num_feature > 25):
                    index = np.random.rand(self.TrainX.shape[1])
                    population[i, index < 0.015] = 1
                    num_feature = np.sum(population[i])
            return  population

       def initialPop(self):
           population = np.zeros((self.popNum, self.TrainX.shape[1]), dtype=float)

           for i in range(50):
               population[i] = self.getValidRow()
           return population


       def getValidRow(self):
           numDescriptors = self.TrainX.shape[1]
           validRow = np.zeros((1, numDescriptors))
           count = 0
           while (count < 5) or (count > 25):
               count = 0
               for i in range(numDescriptors):
                   rand = round(random.uniform(0, 100), 2)
                   if rand < 1.5:
                       validRow[0][i] = 1
                       count += 1
           return validRow
       def isValidRow(self, row):
           count = 0
           for value in row[0]:
               if value == 1:
                   count += 1
           return (count > 5) and (count < 25)

       def initialLocalBestMatrix(self, population, df):

           localBestMatrix = population
           localBestFitness = df.iloc[:, 1].to_numpy()
           return localBestMatrix, localBestFitness

       def updateLocalBestMatrix(self, localBestMatrix, localBestFitness, population, df):
           all_fitness = df.iloc[:,1].to_numpy()
           for i in range(self.popNum):
               #For each row “i” of the current population  If the fitness of the population[i] < fitness of local-best[i]
               if all_fitness[i]< localBestFitness[i] and (0 <= i) and (i < len(population)):
                   localBestMatrix[i] = population[i]
                   localBestFitness[i] = all_fitness[i]
           return localBestMatrix, localBestFitness


           # The initial local best matrix is the same as the initial population to start with

       def updateGlobalBestRow(self, localBestMatrix, localBestFitness,k):
            # The initial global best row is the row in the initial population with the best fitness
            matrix3 = np.concatenate((localBestMatrix,localBestFitness[:, None]), axis=1)
            matrix3 = matrix3[matrix3[:, -1].argsort()]
            new_best_fitness = matrix3[0,-1]
            if self.best_fitness > new_best_fitness:
                self.best_fitness = new_best_fitness
                self.pop_index_best_fitness = k
                print("best fitness:{} is for population number{}".format(self.best_fitness, self.pop_index_best_fitness))
            globalBestRow = matrix3[0,:-1]
            return globalBestRow
            #print("global best row:\n", self.globalBestRow)



#------------------------------------------------------------------------------------------------------------------------------------------
       def modeling(self, population, k):
            #create fitting object(self.fit) from module fitting_scoring
            self.fit = fitting_scoring.fitting()
            #do evaluation process based on regressr(the type of model that was selected) and input cleaned rescaled and splitted data
            self.trackDesc, self.trackFitness, self.trackModel, \
            self.trackDimen, self.trackR2train, self.trackR2valid, \
            self.trackR2test, self.testRMSE, self.testMAE, \
            self.testAccPred = self.fit.evaluate_population(model= self.regressor, instructions=self.instructions, data=self.data,population=population, exportfile='')

            df = self.PrintModelResults(k)
            return df
#------------------------------------------------------------------------------------------------------------------------------------------
       def fitness_sort(self, df):

           #sort Dataframe based on fitness
           df_sort= df.sort_values('Fitness')
           return df_sort
#------------------------------------------------------------------------------------------------------------------------------------------
       def selected_2_total(self, selected_arr):
            # convert selected row to row of population
            total_arr = np.zeros((1, self.TrainX.shape[1]), dtype=float)
            # assigne one to selected_arr
            total_arr[0, selected_arr] = 1
            return total_arr
#-------------------------------------------------------------------------------------------------------------------------------------

       def createNextPopulation(self, alpha, localBestMatrix,velocity, globalBestRow, population):

           # Find the new value of a(it should decrease from 0.5 to 0.33).Therefore, if you do K iterations,
           # each time you need to subtract a = (0.5 - 0.33) / j.

           p = (0.5) * (1 + alpha)
           old_population = population.copy()
           # FindingNewPopulation based on a and p values
           new_population = np.zeros((self.popNum, self.TrainX.shape[1]), dtype=float)
           #velocity = self.updateVelocity(velocity, localBestMatrix, globalBestRow, population)
           for i in range(self.popNum):

               for j in range(self.TrainX.shape[1]):
                   if velocity[i, j] <= alpha:
                       new_population[i, j] = old_population[i, j]
                   elif velocity[i, j] > alpha and velocity[i, j] <= p:
                       new_population[i, j] = localBestMatrix[i, j]
                   elif velocity[i, j] > p and velocity[i, j] <= 1:
                       new_population[i, j] = globalBestRow[j]
                   else:
                       new_population[i, j] = old_population[i, j]
               num_feature = np.sum(new_population[i])
               while (num_feature < 5 or num_feature > 25):
                   new_population[i] = self.getValidRow()
                   num_feature = np.sum(new_population[i])
                   #print("num_features:", num_feature)

           #print("equality of pop1 and pop2 ", np.array_equal(new_population, old_population))
           return new_population
            #print("shape of new pop and old pop ", self.population.shape , self.old_population.shape)
# -------------------------------------------------------------------------------------------------------------------------------------
       def PrintModelResults(self, j):
            #create dataframe based on the dictionary of data that returned from fitting scoring module
            mydicts =[self.trackDesc,self.trackFitness, self.trackModel, self.trackDimen ,self.trackR2train, self.trackR2test, self.trackR2valid, self.testRMSE, self.testMAE, self.testAccPred]
            df = pd.concat([pd.Series(d) for d in mydicts], axis=1).fillna(0).T
            df.index = [ 'Descriptors','Fitness','Model','Dimen','R2train','R2test','R2Validation','RMSE','testMAE','testAccPred']
            df = df.T
            df = df.reset_index(drop=True)
            #assine best fitness of new population to new fitness and compare it to the best fitness of previous population
            #df_sort =  self.fitness_sort(df)
            #new_fitness = df_sort.iloc[0, 1]
            #if (new_fitness < self.best_fitness):
             #   self.best_fitness = new_fitness
              #  self.best_fitness_popNo = j
               # print(" best fitness is {} for population number {}".format(self.best_fitness, self.best_fitness_popNo))
            self.save_to_file(df)
            return df

# -------------------------------------------------------------------------------------------------------------------------------------
       def save_to_file(self,df):
            if os.path.isfile(self.filename):
                # Old data
                #oldFrame = pd.read_csv(self.filename)
                # Concat
                #df_diff = pd.concat([oldFrame, self.df], ignore_index=True).drop_duplicates()
                # Write new rows to csv file
                df.to_csv(self.filename, mode='a', header=False, index= False)
            else:  # else it doesn't exist to  append
                df.to_csv(self.filename, index= False)

# -------------------------------------------------------------------------------------------------------------------------------------
       def removeRedunduntFromFile(self,):
            dFrame = pd.read_csv(self.filename)
            dFrame = dFrame.drop_duplicates()
            dFrame.to_csv(self.filename)




