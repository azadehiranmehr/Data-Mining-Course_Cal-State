
from sklearn import linear_model , svm,svm,neural_network
import pandas as pd
import numpy as np
import random
import fitting_scoring
class Genetic_Algorithm:

        def __init__(self, model,data):
            #classs constructor to initiat a Genetic_Algorithm class
            #based on descriptors and targets file and the model(MLR, SCM or ANN) and number of population that user seleted in the main program
            self.model = model
            if self.model == 0:
                print("you selected:", 'MLR model')
                self.regressor = linear_model.LinearRegression()
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'MLR'}
            elif self.model == 1:
                print("you selected:", 'svm model')
                self.regressor = svm.SVR()
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'SVM'}
            elif self.model == 2:
                print("you selected:", 'ANN model')
                self.regressor = neural_network.MLPRegressor( max_iter=600)
                self.instructions = {'dim_limit': 10, 'algorithm': 'None', 'MLM_type': 'ANN'}
            #copy data that what to run GA for them. data was store in dictionarydata type in previous steps(in process_input module)
            self.data = data
            self.TrainX = self.data['TrainX']
            self.total_feature = self.data['TrainX'].shape[1]
            self.mutation_rate = 0.005
            self.popNum = 50
            self.best_fitness = 100000

        def initial_population_total(self):
            # Set up an initial array of all zeros
            self.population = np.zeros((self.popNum, self.TrainX.shape[1]))
            for i in range(50):
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





        def modeling(self):
            #create fitting object(self.fit) from module fitting_scoring
            self.fit = fitting_scoring.fitting()
            #do evaluation process based on regressr(the type of model that was selected) and input cleaned rescaled and splitted data
            self.trackDesc, self.trackFitness, self.trackModel, \
            self.trackDimen, self.trackR2train, self.trackR2valid, \
            self.trackR2test, self.testRMSE, self.testMAE, \
            self.testAccPred = self.fit.evaluate_population(model= self.regressor, instructions=self.instructions, data=self.data,population=self.population, exportfile='')

        def fitness_sort(self):
            #sort Dataframe based on fitness

            self.df= self.df.sort_values('Fitness')


        def onepoint_crossover(self,parent1,parent2, crossover_point):

            # create child1 and check for correct number of ones between 5 and 25
            child_1 = np.append(parent1[:crossover_point], parent2[crossover_point:])
            num_feature = np.sum(child_1)
            while (num_feature < 5 or num_feature > 25):
                crossover_point = random.randint(1, self.TrainX.shape[1])
                child_1 = np.append(parent1[:crossover_point], parent2[crossover_point:])
                num_feature = np.sum(child_1)
            # create child2 and check for correct number of ones between 5 and 25
            child_2 = np.append(parent2[:crossover_point], parent1[crossover_point:])
            num_feature = np.sum(child_2)
            while (num_feature < 5 or num_feature > 25):
                crossover_point = random.randint(1, self.TrainX.shape[1])
                child_2 = np.append(parent2[:crossover_point], parent1[crossover_point:])
                num_feature = np.sum(child_2)
            return child_1,child_2
        def twopoint_crossover(self,parent1, parent2,crossover_array):
            #create two 2point crossed child over based on 2  sorted random number in the crossover_array
            child_3, child_4 = parent1, parent2
            for i in crossover_array:
                child_3, child_4 = self.onepoint_crossover(child_3, child_4, i)

            return child_3, child_4

        def mutation(self, row):
            mu = 0.0005
            take_mutation = np.random.rand(1)
            if (take_mutation < mu):
                index1 = np.where(row == 1)[0]
                index0 = np.where(row == 0)[0]
                row[np.random.randint(index1.shape[0])] = 0
                row[np.random.randint(index0.shape[0])] = 1
            return row

        def create_next_population(self):
            #sort dataframe based on fitness
            self.fitness_sort()
            self.old_population = self.population
            #dfcol1 : first column of dataframe that stores indexe seperated by _, raplace _ to _ to create a valid indexes
            self.df1col = self.df.iloc[:, 0].apply(lambda x: eval(x.replace('_', ',')))
            # create zero rows for 2 parents
            parent1 = np.zeros((1,self.TrainX.shape[1]))
            parent2 = np.zeros((1,self.TrainX.shape[1]))
            #assigne one to parent1's bits for the index in the first row frist column of the sorted dataframe
            parent1[0, self.df1col.iloc[0]] = 1
            # assigne one to parent2's bits for the index in the first row frist column of the sorted dataframe
            parent2[0, self.df1col.iloc[1]] = 1

            # Choose a random numbers for crossover function
            crossover_point1 = random.randint(1, self.TrainX.shape[1])
            crossover_point2 = random.randint(1, self.TrainX.shape[1])
            #put 2 random numbers in crossover function for 2point crossover function
            crossover_array = np.array([crossover_point1, crossover_point2])
            #sort 2 random numbers to use in 2 pont cross over
            crossover_array = np.sort(crossover_array, axis=0)
            #child1, child2 = self.onepoint_crossover(parent1, parent2, crossover_point1)
            child3, child4 = self.twopoint_crossover(parent1, parent2,crossover_array)
            #create initial population again
            self.initial_population_total()
            #assigne parents an children to those first rows
            self.population[0] = parent1
            self.population[1] = parent2
            #self.population[2] = child1
            #self.population[3] = child2
            self.population[4] = child3
            self.population[5] = child4
            for i in range(50):
                self.population[i] = self.mutation(self.population[i])
            #print("equality of pop1 and pop2 ", np.array_equal(self.population, self.old_population))
        def PrintModelResults(self, j):
            #create dataframe based on the dictionary that returned data from fitting scoring module
            mydicts =[self.trackDesc,self.trackFitness, self.trackModel, self.trackDimen ,self.trackR2train, self.trackR2test, self.testRMSE, self.testMAE, self.testAccPred]
            df = pd.concat([pd.Series(d) for d in mydicts], axis=1).fillna(0).T
            df.index = ['Descriptors','Fitness','Model','Dimen','R2train','R2test','RMSE','testMAE','testAccPred']
            self.df= df.T
            self.df = self.df.reset_index(drop=True)
            #assine best fitness of new population to new fitness and compare it to the best fitness of previous population
            self.fitness_sort()
            self.new_fitness = self.df.iloc[0, 1]
            if (self.new_fitness < self.best_fitness):
                self.best_fitness = self.new_fitness
                self.best_fitness_popNo = j
                print(" best fitness is {} for population number {}".format(self.best_fitness, self.best_fitness_popNo))
            #print(" dataframe--------------------------:\n", self.df)
            if (self.model == 0):
                self.df.to_csv("mlr.csv")
            elif (self.model == 1):
                self.df.to_csv("svm.csv")
            else:
                self.df.to_csv("ann.csv")



