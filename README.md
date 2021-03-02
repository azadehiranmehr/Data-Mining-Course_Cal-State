# Data-Mining-Course_Cal-State
In the Datamining course I have done several labs and 4 projects  
Project 1: Doing data Modeling using different machine Learning:
MLR: Multiple Linear Regression
SVM: Support Vector Machin
ANN: Artificial neural network

You are required to complete this project, writing the codes that have not been implemented and run the program with three different machine learning tools. There are three files
-	main_datamining.py
-	fitting_scoring.py 
-	process_input.py
In the fitting_scoring.py you should implement
-	the fitness function and 
-	calculateRMSE() function
In the process_input.py you should implement
-	sort_descriptor_matrix
-	open_descriptor_matrix function
-	open_taget_values function and 
-	Remove_invalid_rows function

Run the work with three machine learning model (MLR, SVM and ANN). 

The fitness function is implemented as:
numerator = ((mt - n - 1 ) * RMSEt2 )+ (mv*RMSEv2)
denominator = mt - n -1 + mv
f = Sqrt (numerator / denominator)

•	mt is the number of samples of training. 
•	mv is the number of samples of validation
•	n is the total number of descriptors
•	RMSEv is the Root mean square error of validation: 
•	RMSEt is the Root mean square error of training 

Test your program with some small data set first to see if you are getting correct result. Then test it with the given two data sets files, and take the snapshot of three outputs: 
-	Run it with MLR, get the snapshot of the results and place it in a file called Project-1-MLR
-	Run it with SVM, get the snapshot of the results and place it in a file called Project-1-SVM
-	Run it with ANN, get the snapshot of the results and place it in a file called Project-1-ANN

----------------------------------------------------------------------------------------------------------------------------------------
Project 2: Doing data Modeling using Genetic Algorithm with:
MLR: Multiple Linear Regression
SVM: Support Vector Machin
ANN: Artificial neural network

You are required to add expand project 1 and enhance it with Genetic Algorithm technique. You should view and listen to the provided video and see how it is supposed to be done.

Run the work with three machine learning model (MLR, SVM and ANN). The results should go into three output files: 
-	MLR_Output.csv 
-	SVM_Output.csv
-	ANN_Output.csv

-	Look at the each file and based on the values of fitness (The closer to zero, the better it is. Negative number is no good at all) , R2 of training and R2 of validation and R2 of testing (All should be close to each other with around 0.1 different and should be > 0.5 and <1) and number of dimensions and RMSE fitness (The closer to zero, the better it is. Negative number is no good at all), decide which result is the best one that you would choose. Sort the results based on what you think is the best result. 
------------------------------------------------------------------------------------------------------------------------------------------
Project 3: Doing data Modeling using Differential Evolution (DE) Algorithm with:
MLR: Multiple Linear Regression
SVM: Support Vector Machin
ANN: Artificial neural network

You are required to expand project 1 and enhance it with Differential Evolutionary (DE) Algorithm technique. You should view and listen to the provided video and see how it is supposed to be done.

Run the work with three machine learning model (MLR, SVM and ANN). The results should go into three output files: 
-	MLR_Output.csv 
-	SVM_Output.csv
-	ANN_Output.csv
-	
-	Look at each output file and based on the values of fitness (The closer to zero, the better it is. Negative number is no good at all) , R2 of training and R2 of validation and R2 of testing (All should be close to each other with around 0.1 different and should be > 0.5 and <1) and number of dimensions and RMSE fitness (The closer to zero, the better it is. Negative number is no good at all), decide which result is the best one that you would choose. Sort the results based on what you think is the best result. 

----------------------------------------------------------------------------------------------------------------------------------------------
Project 4: Doing data Modeling using Binary particle Swarm Optimization (BPSO) with:
MLR: Multiple Linear Regression
SVM: Support Vector Machin
ANN: Artificial neural network

You are required to expand project 1 and enhance it with Binary particle Swarm Optimization (BPSO) Algorithm technique. You should view and listen to the provided video and see how it is supposed to be done.

Run the work with three machine learning model (MLR, SVM and ANN). The results should go into three output files: 
-	MLR_Output.csv 
-	SVM_Output.csv
-	ANN_Output.csv
-	
-	Look at each output file and based on the values of fitness (The closer to zero, the better it is. Negative number is no good at all) , R2 of training and R2 of validation and R2 of testing (All should be close to each other with around 0.1 different and should be > 0.5 and <1) and number of dimensions and RMSE fitness (The closer to zero, the better it is. Negative number is no good at all), decide which result is the best one that you would choose. Sort the results based on what you think is the best result. 

