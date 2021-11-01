# Project1
# Machine learning project done by Justine Montavon Kevin Rizk, and Louis  Auzeau
This project was done for the project 1 of "EPFL Higgs boson challenge". It will use machine learning techniques to train a function on the train data and then make a prediciton on the test data set.

Here is a guideline (subjective) and how to proceed and run the program to make the predictions file. To be able to run the program you need to have python 3 installed. Python 2.7 will not work and python 3.8 could work but was not tested.

The code is documented if you want to make any change or use another fucntion or to understand what each function does.
If python3 and numpy are already installed on your machine you can jump to Step 4

#Step 1: open a command window. 

#Step 2: check that python is installed on you machine by running the command "python"

#Step 3: install the module numpy by running the command "pip install numpy"

#Step 4: Navigate to the data folder and extract the zip files of train.csv and test.csv. Make sure that the csv files are in the folder named "data"

#Step 5: Open the terminal or the cmd (windows) and position yourself in the folder where the file run.py is located using the commend "cd your_folder_Path" for example

#Step 6: Run on the cmd or terminal the command "python3 run.py" or simply "python run.py". The code will run and will let you know at which step you are as the code is running. There are four indexes, ranging from 0 to 3 and degrees goes from 4 to 18. Each value of index correspond to one of the four group that compose our data.
When the code has finished running on an index it will display several informations, especially the minimal error on this index and the degree and lambda associated. 
When it is finished, it will display the message "FINISHED", the final error on our train set and produce a file named "final_submission.csv". This file contains our prediction for the test file.



