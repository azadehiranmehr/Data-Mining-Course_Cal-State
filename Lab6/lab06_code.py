def mutation(self, row):
    # mu:mutation rate
    mu = 0.0005
    # take a random number for each index in the row (mu_index contains random number for each index of a row)
    mu_index = np.random.rand(self.TrainX.shape[1])
    # Flip each bit of the row if the mu_index for that index is <= mu
    row[np.logical_and(mu_index <= mu, row == 0)] = 1
    row[np.logical_and(mu_index <= mu, row == 1)] = 0
    return row


def create_next_population(self):
    self.fitness_sort()
    self.old_population = self.population
    self.df1col = self.df.iloc[:, 0].apply(lambda x: eval(x.replace('_', ',')))

    parent1 = np.zeros((1, self.TrainX.shape[1]))
    parent2 = np.zeros((1, self.TrainX.shape[1]))

    parent1[0, self.df1col[0]] = 1
    parent2[0, self.df1col[1]] = 1

    # Choose a random number of ones to create
    crossover_point1 = random.randint(1, self.TrainX.shape[1])
    crossover_point2 = random.randint(1, self.TrainX.shape[1])
    crossover_array = np.array([crossover_point1, crossover_point2])
    crossover_array = np.sort(crossover_array, axis=0)
    child1, child2 = self.onepoint_crossover(parent1, parent2, crossover_point1)
    child3, child4 = self.twopoint_crossover(parent1, parent2, crossover_array)
    self.initial_population_total()

    self.population[0] = parent1
    self.population[1] = parent2
    self.population[2] = child1
    self.population[3] = child2
    self.population[4] = child3
    self.population[5] = child4
    for i in range(50):
        self.population[i] = self.mutation(self.population[i])