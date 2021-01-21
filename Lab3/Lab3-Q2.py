import pandas as pd
import numpy as np
class myMatrix:
    def __init__(self, n,fileName):
        self.n = n
        self.fileName = fileName
        content = np.loadtxt(self.fileName , delimiter = '\t',dtype = 'int32')
        matrixSize = content.reshape(-1).size
        if self.n**2 > matrixSize :
            print("Size of matrix is less than your number please enter another number:")
            exit()
        self.matrix = content.reshape(-1)[:self.n**2].reshape(self.n,self.n)
    @staticmethod
    def Product(M1,M2):
        productMatrix =np.multiply(M1, M2)
        return productMatrix

    @staticmethod
    def DotProduct(M1, M2):
        dotMatrix =np.dot(M1, M2)
        return dotMatrix

    @staticmethod
    def Division(M1, M2):
        divideMatrix = np.divide(M1,M2)
        return divideMatrix

#get a size for a square matrix
n = int(input("input a number for size of square matrix:  "))
if n <= 3 or n >= 8:
   print("please enter another number, this number is out of range")
   exit()

#Create an object call it myMatrix
M1 = myMatrix(n, 'file1.txt')
M2 = myMatrix(n, 'file2.txt')
M3 = myMatrix(n, 'file3.txt')
print("\nmatrix1:\n", M1.matrix)
print("matrix2:\n", M2.matrix)
M1_Multiply_M2 = myMatrix.Product(M1.matrix, M2.matrix)
print("\nMatrix1 Multiply Matrix2 is:\n", M1_Multiply_M2)
M1_DotMultiply_M2 = myMatrix.DotProduct(M1.matrix, M2.matrix)
print("\nMatrix1 Dot Product Matrix2 is:\n", M1_DotMultiply_M2)
M1_Divde_M2 = myMatrix.Division(M1.matrix, M2.matrix).round(2)
print("\nMatrix1 Divede Matrix2 is:\n",M1_Divde_M2 )#(show it with at least 2 significant digits)
#y = pd.DataFrame(M1_Divde_M2)
#y[y == np.inf] = 'undef'
#print("\nreplacing inf to undef if there is any inf(divide by zero)\n", y)
#Now change the content of the second matrix as follows (the changes
print("\n********************Second part:Outputs when replacing some values of second matrix with Zero(second matrix is matrix3)****************\n ")
M1_Multiply_M3 = myMatrix.Product(M1.matrix, M3.matrix)
print("\nMatrix1 Multiply matrix3 is:\n", M1_Multiply_M3)
M1_DotMultiply_M3 = myMatrix.DotProduct(M1.matrix, M3.matrix)
print("\nMatrix1 Dot Product Matrix3 is:\n", M1_DotMultiply_M3)
M1_Divde_M3 = myMatrix.Division(M1.matrix, M3.matrix).round(2)
#np.around(M1_Divde_M3, 2,inplace='True')
print("\nMatrix1 Divede Matrix3 is:\n",M1_Divde_M3)#(show it with at
y = pd.DataFrame(M1_Divde_M3)
y[y == np.inf] = 'undefined'
print("\nreplacing inf to undefined if there is any inf(divide by zero):\n", y)
