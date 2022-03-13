# Neural_Network_Charity_Analysis

## Overview 
The purpose of this project is to assist my client (Alphabet Soup) predict where to make investments using machine learning and neural networks.  This effort requires me to use the features in the provided dataset to help Alphabet Soup create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.


I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

    - EIN and NAME—Identification columns
    - APPLICATION_TYPE—Alphabet Soup application type
    - AFFILIATION—Affiliated sector of industry
    - CLASSIFICATION—Government organization classification
    - USE_CASE—Use case for funding
    - ORGANIZATION—Organization type
    - STATUS—Active status
    - INCOME_AMT—Income classification
    - SPECIAL_CONSIDERATIONS—Special consideration for application
    - ASK_AMT—Funding amount requested
    - IS_SUCCESSFUL—Was the money used effectively

As always, my analysis followed the data analysis principles of (1) Determine the number of rows and columns; (2) Data types used; and (3) Is the data readable?

__Client Deliverables:__
- Deliverable 1: Preprocessing Data for a Neural Network Model (Data Processing)
- Deliverable 2: Compile, Train, and Evaluate the Model
- Deliverable 3: Optimize the Model

## Resources
The resouces used for this analysis included;
- Dataset: ![Charit Dataset](https://github.com/SheaButta/Amazon_Vine_Analysis/blob/main/Images/charity_data.csv)
- Machine Learning
- Neural Networks
- Visual Studio Code
- Python
- Pandas
- Skikit learn
- StandardScaler
- Tensorflow


## Results
All sprints were completed as scheduled and I delivered on all client expectations/results. The below images will visualize the expected client results.

### Deliverable 1: Preprocessing Data for a Neural Network Model (Data Processing)

To preprocess the data for the Neural Network Model the following tasks were completed;

#### The EIN and NAME columns have been dropped

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_1_dropEinName.PNG)


#### The columns with unique values have been grouped together

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_1_uniqColumnsGrouping.PNG)


#### The categorical variables have been encoded using one-hot encoding

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_1_uniqColumnsGrouping.PNG)


#### The images shows the follwoing;

 - Preprocessed data is split into features and target arrays
 - The data is split into training and testing datasets 
 - The numerical values have been standardized using the StandardScaler() module

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_1_uniqColumnsGrouping.PNG)



### Deliverable 2: Compile, Train, and Evaluate the Model

This deliverable required me to generate answers/conclusions to a few important topics.

#### The number of layers, the number of neurons per layer, and activation function are defined
#### An output layer with an activation function is created
#### There is an output for the structure of the model
#### There is an output of the model’s loss and accuracy
#### The model's weights are saved every 5 epochs 
#### The results are saved to an HDF5 file


There are three layers; two (2) hidden and one (1) outer layer.  This deep-learning neural network model is made of two hidden layers with 80 and 30 neurons.  The first and second layers use the "relu" activation.  There is also an outer layer which  utilizes the "sigmoid" activation.

   ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_2_Layers.PNG)

Output for the structure of the model:

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_2_OutputModelStructure.PNG)


Output of the model’s loss and accuracy:

    ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_2_ModelLossAccuracy.PNG)


### Deliverable 3: Optimize the Model

An optimized model was created with the following;

#### Noisy variables are removed from features
#### Additional neurons are added to hidden layers
#### Additional hidden layers are added
#### The activation function of hidden layers or output layers is changed for optimization
#### The model's weights are saved every 5 epochs
#### The results are saved to an HDF5 file

   
   ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_3_OptimizedLayers.PNG)

   ![](https://github.com/SheaButta/Neural_Network_Charity_Analysis/blob/main/Images/Deliv_3_OptimizedModelLossAccuracy.PNG)

  
## Summary
The project has been completed.  The work neural network model did not reach the target of 75% accuracy. The target level would seem to indicate the model is not outperforming.
We could ultimately use a supervised machine learning model such as the Random Forest Classifier to combine various decision trees and generate a classified output and evaluate its performance against our deep learning model.  This is all due to using a binary classification.
