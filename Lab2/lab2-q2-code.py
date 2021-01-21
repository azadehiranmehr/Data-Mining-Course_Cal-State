
import pandas as pd
from openpyxl.workbook import Workbook

#load excel file into pandas dataframe
df = pd.read_excel('AlzheimerData.xlsx', skiprows=1)
#original dataframe
print("original dataframe:\n:", df)


#Remove all columns with zero value
df = df.loc[:, (df != 0).any(axis=0)]
print("\nDataframe after removing column with all zero:\n", df)


#check type of each coulum and then count of coulmn for each type
print("\ncheck type of each coulum and then count of coulmn for each type:\n", df.dtypes.value_counts())


#dfo is the part of dataframe that contain garbage characters and should be clean
dfo = df.select_dtypes('object')
print("\nthe part of dataframe that contain garbage characters and should be clean:\n", dfo)



#show total number of each type in the part of dataframe containig garbage(dfo :the part that we want to clean).
#print("\ntotal number of each type in the part of dataframe containig garbage:\n", dfo.applymap(type).stack().value_counts())
#Parse function: input x(cell of the daraframe), return True if length of x equal and greater than 5 and False if less than 5
# this function is used to check whether we should drop a row or coulmn(if there is a string more than 5, they should be dropped)
def parse(x):
    if type(x)== str:
        import re
        x_str=re.sub(x,'[0-9]','')
        if len(x_str)>=5:
            return  True
    return False

#Remove all rows and columns if they have string more than 5 by the use of pars function to check it
#dfo2 keep rows and columns of dfo that all cells of that row or column contain either int or float or string less than 5
#So when we apply dfo.applymap(parse) all cells get false or 0,(for rows sum(1)==0) and for coulumn sum(0)==0)
dfo22= dfo.applymap(parse)

print("\ndfo22:\n",dfo22)

#Remove all rows and columns if they have string more than 5 by the use of pars function to check it
#dfo2 keep rows and columns of dfo that all cells of that row or column contain either int or float or string less than 5
#So when we apply dfo.applymap(parse) all cells get false or 0,(for rows we can set sum(1)==0 before ,) and for coulumn sum(0)==0 after ,)
dfo2= dfo.loc[:,dfo.applymap(parse).sum(0)==0]
print("\n after removing column with string equal and greater than 5 from dfo:\n", dfo2)




#Parse_replace: replce each cell of dataframe by float number except for  error
# for string except raise and replace string with 0
#these strings should be less than 5 because all string equal and greater than 5 have alrealy been deleted in previous part
def parse_replace(x):
    try:
        return float(x)
    except:
        return 0

df_clean = dfo2.applymap(parse_replace)
print("\n after cleaning dfo:\n", df_clean)


#Clean part of dataframe that exclude from dirty part of dataframe in the the begining
df_exclude = df.select_dtypes(exclude='object')

print("\nexclude:\n", df_exclude)

#Concat dfo2(the part of dataframe that got cleaned) with the rest clean part of data frame and fill nan values with zero
df2 = pd.concat([df_exclude, df_clean], axis=1).fillna(0)
print("df2\n:", df2)
#reorder column order based on original column order of dataframe(swap column X1 and X2)
cols = list(df2.columns)
a, b = cols.index('X1'), cols.index('X2')
cols[b], cols[a] = cols[a], cols[b]
df2 = df2[cols]

#final clean dataframe
print("\n final clean dataframe:\n", df2)

#df3 Dataframe with only X values (Remove column Y)
df3 = df2.iloc[:,1:]
# finding std of whole X columns
std = df3.stack().std(skipna = True)
print("\ntotal standard deviation is:\n",std)

# finding mean of whole X columns
xmean= df3.mean().mean()
print("\ntotal mean is:\n",xmean)
#finding minimum of whole X Columns
xmin = df3.min().min()
print("\ntotal min is:\n",xmin)
#finding minimum of whole X Columns
xmax = df3.max().max()
print("\ntotal max is:\n",xmax)

#Calulate standardization( transforms data to have a mean of zero and a standard deviation of 1)
#Calculate normalization(scale a variable to have a values between 0 and 1)

x_standardization=lambda x: (x-xmean)/std
x_normalization=lambda x: (x-xmin)/(xmax-xmin)
#df2.apply(x_standardization).T.dropna(how='all').T

#apply x_standardization to clean dataframe (df2)
standardization = df3.loc[:,df3.std()>0].apply(x_standardization)
print("\n after applying standardization to clean dataframe:\n", standardization)


#apply x_normalization to the clean dataframe(df2)
normalization = df3.loc[:,df3.std()>0].apply(x_normalization)
print("\nafter applying normalization to clean dataframe:\n", normalization)



standardization.to_csv("Lab2-Q2-Rescaled.csv")
normalization.to_csv("Lab2-Q2-Normalized.csv")
df2.to_csv("Final-Clean-Data.csv")
