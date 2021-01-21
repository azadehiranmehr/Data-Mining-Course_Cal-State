import numpy as np
import pandas as pd

data1 = np.loadtxt('data.txt', delimiter=',', dtype = 'int')

print(type(data1))
print(data1.dtype)
 #print the array
print("original matrix is : \n", data1)

'''	Select three columns, columns 3, 1, and 9 and create filterindex matrix to
be used for filtering original matix' columns and select only columns 3,1 and 9'''
filterindex =  [True,False,True,False,False,False,False,False,True,False]
filterarray = data1[ : ,filterindex]
print("\nfiltered array ,Show Column 1,3, 9 of original matix:\n")
print(filterarray)
'''Reaggarnge the order of the columns based on permutation list order'''
permutation = [1, 0, 2]
idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
filterarray[:, idx]  # return a rearranged copy
filterarray[:] = filterarray[:, idx]  # in-place modification of arearranged matrix
matrix1 = filterarray
print("\nmatrix1 : filterd and rearranged original matrix: to the order of column 3, 1 and 9\n")
print(matrix1)
'''sort the filtered and rearranged matrix'''
print("\n The sorted version of Matrix1 in ascending order of its first column (column 3 of the original matrix) is:\n")

matrix1 = filterarray[filterarray[:,0].argsort()] # column 3 of original matrix is first column (with index 0)of new matrix
print(matrix1)

print("\n *********** end of answer to Question 2 part b: ***************")
filterindex2 =  [False,True,False,False,True,False,True,False,False,False]
filterarray2 = data1[ : ,filterindex2]
print("\nfiltered array, show column 2,5 and 7\n")
print(filterarray2)
permutation = [1, 0, 2]
idx = np.empty_like(permutation)
idx[permutation] = np.arange(len(permutation))
filterarray2[:, idx]  # return a rearranged copy
filterarray2[:] = filterarray2[:, idx]  # in-place modification of arearranged matrix
matrix2 = filterarray2
print("\nMatrix2: filterd and rearranged matix: to the order of column 5, 2 and 7\n")
print(matrix2)

#sort filtered and rearranged matrix
print("\nSorting Matrix2 in descending order of its first column (column 5 of the original matrix) we get:\n")
matrix2 = filterarray2[filterarray2[:,0].argsort()[::-1]]
print(matrix2)
#Add two matrix together
matrix3 = np.add(matrix1 , matrix2)
print("\nMatrix3: sum of above 2 sorted matrix(matrix1 and matrix2)\n")
print(matrix3)
#Sum rows togheter and creat one column
matrix4 = np.sum(matrix3, axis=1)
print("\nMatrix4: which each number is sum of all rows of matrix3\n")
print(matrix4)
# Sort the last matrix in
matrix4 = np.sort(matrix4)
print("show sort order of matrix4 vertically:\n ", matrix4.reshape(-1, 1)) #reshape last matrix vertically
