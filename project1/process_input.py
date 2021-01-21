from numpy import *
import pandas as pd
import csv

# ------------------------------------------------------------------------------------------------

def rescale_data(descriptor_matrix):

    # You have already written code for this.
    xmean = descriptor_matrix.mean().mean()
    std = descriptor_matrix.stack().std()
    #Calulate standardization( transforms data to have a mean of zero and a standard deviation of 1)
    x_standardization=lambda x: (x-xmean)/std
    #apply x_standardization to clean dataframe
    standardization = descriptor_matrix.loc[:, descriptor_matrix.std() > 0].apply(x_standardization)
    return standardization

# ------------------------------------------------------------------------------------------------
# What do we need to sort the data?

def sort_descriptor_matrix(descriptors, targets):
    # Placing descriptors and targets in ascending order of target (IC50) value.
    alldata = ndarray((descriptors.shape[0], descriptors.shape[1] + 1))
    alldata[:, 0] = targets
    alldata[:, 1:alldata.shape[1]] = descriptors
    alldata = alldata[alldata[:, 0].argsort()]
    descriptors = alldata[:, 1:alldata.shape[1]]
    targets = alldata[:, 0]

    return descriptors, targets

# ------------------------------------------------------------------------------------------------

# Performs a simple split of the data into training, validation, and testing sets.
# So how does it relate to the Data Mining Prediction?

def simple_split(descriptors, targets):

    testX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 0]
    validX_indices = [i for i in range(descriptors.shape[0]) if i % 4 == 1]
    trainX_indices = [i for i in range(descriptors.shape[0]) if i % 4 >= 2]

    TrainX = descriptors[trainX_indices, :]
    ValidX = descriptors[validX_indices, :]
    TestX = descriptors[testX_indices, :]

    TrainY = targets[trainX_indices]
    ValidY = targets[validX_indices]
    TestY = targets[testX_indices]

    return TrainX, ValidX, TestX, TrainY, ValidY, TestY

# ------------------------------------------------------------------------------------------------

# try to optimize this code if possible

def open_descriptor_matrix(fileName):
   ''' preferred_delimiters = [';', '\t', ',', '\n']

    with open(fileName, mode='r') as csvfile:
        # Dynamically determining the delimiter used in the input file
        row = csvfile.readline()

        delimit = ','
        for d in preferred_delimiters:
            if d in row:
                delimit = d
                break

        # Reading in the data from the input file
        csvfile.seek(0)
        datareader = csv.reader(csvfile, delimiter=delimit, quotechar=' ')
        dataArray = array([row for row in datareader if row != ''], order='C')

    if (min(dataArray.shape) == 1):  # flatten arrays of one row or column
        return dataArray.flatten(order='C')
    else:
        return dataArray'''
   dfd = pd.read_csv('Practice_Descriptors.csv', header=None)
   return dfd
#************************************************************************************
#Try to optimize this code if possible

def open_target_values(fileName):
    '''preferred_delimiters = [';', '\t', ',', '\n']

    with open(fileName, mode='r') as csvfile:
        # Dynamically determining the delimiter used in the input file
        row = csvfile.readline()
        delimit = ','
        for d in preferred_delimiters:
            if d in row:
                delimit = d
                break

        csvfile.seek(0)
        datalist = csvfile.read().split(delimit)
        if ' ' in datalist:
            datalist = datalist[0].split(' ')

    for i in range(datalist.__len__()):
        datalist[i] = datalist[i].replace('\n', '')
        try:
            datalist[i] = float(datalist[i])
        except:
            datalist[i] = datalist[i]

    try:
        datalist.remove('')
    except ValueError:
        no_empty_strings = True

    return datalist
'''
    dft = pd.read_csv('Practice_Targets.csv', header=None)
    return dft
#**********************************************************************************************
# Removes constant and near-constant descriptors.
# But I think also does that too for real data.
# So for now take this as it is

def removeNearConstantColumns(data_matrix, num_unique=10):

    useful_descriptors = [col for col in range(data_matrix.shape[1])
                          if len(set(data_matrix[:, col])) > num_unique]
    filtered_matrix = data_matrix[:, useful_descriptors]

    remaining_desc = zeros(data_matrix.shape[1])
    remaining_desc[useful_descriptors] = 1

    return filtered_matrix, where(remaining_desc == 1)[0]

# ------------------------------------------------------------------------------------------------


# part 1: Removes all rows with junk (ex: NaN, etc). Note that the corresponding IC50 value should bedeleted too
# Part 2: Remove columns with 20 junks or more. Otherwise the junk should be replaced with zero
# Part 3: remove all columns that have zero in every cell
def parse_check(x):
    try:
        float(x)
        return False
    except:
        return True


def parse_replace(x):
    try:
        return float(x)
    except:
        return 0

def removeInvalidData(descriptors, targets):
    # Write your code in here
    #concat targets and descriptors to do cleaning on rows
    df = pd.concat([descriptors,targets], ignore_index=True, axis=1)
    #rows cleaning:# part 1: Removes all rows with junk (ex: NaN, etc). Note that the corresponding IC50 value should bedeleted too
    #delete each rows than contains only zero
    df = df.loc[(df.iloc[:,1:]!=0).any(axis=1),:]
    #convert the values in the dataset into a float format for errors put Nan values
    df = df.apply (pd.to_numeric, errors='coerce')
    #delet rows which have all NaN
    df = df.dropna(how='all')
    targets = df.iloc[:,0]
    #Firstly if sum of jucnk is more than 20 those columns will be romoved (with parse_check())
    #Secondly for the remaining dataframe convert each value to float number for the errors replace 0(with parse replace())
    descriptors.loc[:, descriptors.applymap(parse_check).sum(0) <= 20].applymap(parse_replace)
    descriptors = descriptors.loc[:, (descriptors != 0).any(axis=0)]

    return descriptors, targets
