
import pandas as pd
import numpy as np
from openpyxl.workbook import Workbook


dataframe1 = pd.read_excel('myGamma2.xlsx')


# Shape of original Dataframe
dataframe1.shape



df = dataframe1.copy()



#replace all blank character('',' ','  ',...) to Nan
df.replace(r'\s+', np.nan, regex=True, inplace = True)



# check the total number of null value for NAME column
print("\ncheck the total number of null value for NAME column\n", df['NAME'].isna().sum())


#total number of molecules with no IC50 that are deleted
print("\ncheck the total number of null value for IC50 (nM) column\n",df['IC50 (nM)'].isna().sum())



#Any molecule with no IC50 value should be deleted
#df = df[df['IC50 (nM)'].notnull()]
df = df.dropna(subset=('IC50 (nM)',)).reset_index(drop=True)
#Any molecule with no name should be deleted
#df = df[df['NAME'].notnull()]
df = df.dropna(subset=('NAME',)).reset_index(drop=True)




#check that there is no  molecules with no value for IC50
df['IC50 (nM)'].isna().sum()
print("\ncheck the total number of null value for IC50 (nM) column after cleaning of this column\n", df['IC50 (nM)'].isna().sum())

# check the total number of null value for NAME column
df['NAME'].isnull().sum()
print("\ncheck the total number of null value for NAME column after cleaning Null of this column\n", df['NAME'].isna().sum())




#Parse_replace: try cast each cell of dataframe to float number
# for string(garbage) that  we got error (in except part), we return True and replace string with 0
#these strings should be less or equal to 200 because all string greater than 200 have alrealy been deleted in previous part
def parse_replace(x):
    try:
        float(x)
        return False
    except:

        return True
def parse_replace2(x):
    try:
        return float(x)
    except:

        return 0

dfo2 =df.loc[:, df.applymap(parse_replace).sum(axis=0)<=200 ].applymap(parse_replace2)


#total junk column with more than 200 junc that has been deleted
y=df.shape[1] - dfo2.shape[1]
print("\nTotal number of Junk coulmns that should be deleted:\n",y)



#Total number of descriptors (columns) that sum to zero (Number of Zero coulumns that should be deleted)
print("\nTotal number of Zero coulmns that should be deleted:\n",dfo2.loc[:, (dfo2== 0).all(axis=0)].shape[1])


#get rid of the columns that are all zeros
df3 = dfo2.loc[:, (dfo2 != 0).any(axis=0)]


#check that total number of descriptors (columns) after deleting all zero columns in previous step. There should be no Zero column.
print("\nMake sure all zero columns were deleted(total number of zero coulmn):\n",df3.loc[:, (df3== 0).all(axis=0)].shape[1])


#Shape of Dataframe after cleaning
print("\nShape of Dataframe after cleaning:\n",df3.shape)
print("\n Final Data after cleaning:\n", df3)
#select Xes only for rescaling
df4=df3.iloc[:,1:]


#finding mean of whole X coulums of dataframe
xmean = df4.mean().mean()
print("\ntotal standard deviation is:\n",xmean)


# finding STD of whole X columns of dataframe
std = df4.stack().std(skipna = True)
print("\ntotal standard deviation is:\n",std)

#Calulate standardization( transforms data to have a mean of zero and a standard deviation of 1)
#Calculate normalization(scale a variable to have a values between 0 and 1)
x_standardization=lambda x: (x-xmean)/std

#apply x_standardization to clean dataframe (df2)
standardization = df4.loc[:,df4.std()>0].apply(x_standardization)
print("\napply x_standardization to clean dataframe:\n",standardization)

df3.to_csv("Clean-Data-lab4.csv")
df3.to_csv("rescale-Data-lab4.csv")
