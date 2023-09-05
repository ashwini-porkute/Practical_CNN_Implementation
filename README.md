# Practical_CNN_Implementation
![](https://www.google.com/url?sa=i&url=https%3A%2F%2Fsaturncloud.io%2Fblog%2Fa-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way%2F&psig=AOvVaw2ZI3yZK4iZXWiMuNuMnSLy&ust=1693993063536000&source=images&cd=vfe&opi=89978449&ved=0CA4QjRxqFwoTCOj0-YqWk4EDFQAAAAAdAAAAABAD)

CNN model implemented for cifar10 dataset multiclass classification problem.

CNN.py Script reference taken from "https://www.tensorflow.org/tutorials/images/cnn"

# Steps to be followed while designing CNN:
--------------------------------------------

Step 1: Importing basic libraries required
-------
- tensorflow for model api and keras layers requirement.
- matplotlib used to plot the visualizing diagrams
- pandas used for access/alter dataset
- sklearn to divide the dataset, scaling the features of dataset.

Step 2: Reading dataset and splitting
-------
Reading the cifar10 image dataset using tensorflow inbuilt load_data function(dataset loading), it also divides the data into train and test.

Step 3: Feature Engineering
-------
Feature engineering as follows:
- scaling the pixel values of image to the maximum pixel value i.e 255

Step 4: Visualizing the data
-------
Plotting the images using matplotlib for EDA(Exploratory Data Analysis)

Step 5: Developing the model
-------
Creating the CNN model
- create and stacked "Conv2D layer followed with MaxPooling2D", flattened the output and then created fully connected layers for output.

Step 6: Compile and train Model
-------
Compiling and training the CNN model.

Step 7: Evaluating the Trained model
-------
Plotted the graph of train and validation accuracy throughout the training.

Step 8: Inferencing on the model
-------
Performing prediction on trained model and evaluated the performance metrics.
