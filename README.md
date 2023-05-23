# Practical_CNN_Implementation
CNN model implemented for cifar10 dataset multiclass classification problem.

CNN.py Script reference taken from "https://www.tensorflow.org/tutorials/images/cnn"

# Steps to be followed while designing CNN:
--------------------------------------------

Step 1:
-------
Importing basic libraries required
- tensorflow for model api and keras layers requirement.
- matplotlib used to plot the visualizing diagrams
- pandas used for access/alter dataset
- sklearn to divide the dataset, scaling the features of dataset.

Step 2:
-------
Reading the cifar10 image dataset using tensorflow inbuilt load_data function(dataset loading), it also divides the data into train and test.

Step 3:
-------
Feature engineering as follows:
- scaling the pixel values of image to the maximum pixel value i.e 255

Step 4:
-------
Plotting the images using matplotlib for EDA(Exploratory Data Analysis)

Step 5:
-------
Creating the CNN model
- create and stacked "Conv2D layer followed with MaxPooling2D", flattened the output and then created fully connected layers for output.

Step 6:
-------
Compiling and training the CNN model.

Step 7:
-------
Plotted the graph of train and validation accuracy throughout the training.

Step 8:
-------
Performing prediction on trained model and evaluated the performance metrics.
