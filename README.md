# Hepatic_vessels_and_Tumors_Detector
Hepatic Vessel Segmentation using nnU-Net
This repository contains code for segmenting hepatic vessels using the nnU-Net framework. The code is designed to run on Google Colab, utilizing GPU acceleration for deep learning tasks.

Prerequisites
Before running the code, ensure you have the following prerequisites installed:
•	Python 3.x
•	Google Colab (if running on the cloud)
•	CUDA-compatible GPU and drivers (if running locally)

Project Structure
•	nnUnet_HepaticVessels.ipynb: Main Jupyter notebook containing all the steps to train and test 
the model.
•	data/: Directory to store training and testing data.
•	models/: Directory to save trained models.
•	results/: Directory to save the segmentation results.

Setup Instructions
1.	Checking Resources First, check the CUDA environment to ensure proper GPU setup.
2.	Mount Google Drive (Colab Users Only) Mount Google Drive to access data and save results.
3.	Set Base Directory Specify the base directory for the project.
4.	Prepare the Environment Install necessary packages.
5.	Import Libraries Import all required libraries.
6.	Prepare Directories Create necessary directories for data, models, and results.
7.	Data Preparation Load and preprocess the data. Ensure the data is in the correct format for nnU-Net.
8.	Model Training Train the nnU-Net model on the dataset. Customize the training parameters as needed.
9.	Model Evaluation Evaluate the trained model on the test dataset. Visualize the results using matplotlib.

How to Run the Code
1.	Open Google Colab Go to Google Colab.
2.	Upload the Notebook Upload the nnUnet_HepaticVessels.ipynb file to Colab.
3.	Run the Cells Execute the notebook cells in order. Make sure to follow the setup instructions to prepare the environment and data.
4.	Train and Evaluate Follow the steps in the notebook to train and evaluate the model. Customize any parameters as needed for the specific use case.
5.	Visualize Results Use the provided code to visualize the segmentation results.

____

CNN U-Net for Hepatic Vessel Segmentation
This Jupyter notebook (cnnUnet_HepaticVessels.ipynb) demonstrates the implementation of a Convolutional Neural Network (CNN) using the U-Net architecture for the purpose of segmenting hepatic vessels from medical images. The notebook is structured to be run on Google Colab, utilizing Google Drive for data storage and retrieval.

Setup and Installation
1.	Mount Google Drive The notebook starts by mounting Google Drive to access data stored in Google Drive.
2.	Install Required Libraries Install necessary Python packages such as nibabel for handling medical imaging data and tensorflow for building and training the model.
3.	Import Libraries All essential libraries are imported for data manipulation, model building, and evaluation.
4.	Data Preparation This section involves loading and preprocessing the medical imaging data. Ensure the data is correctly structured in Google Drive.
5.	Model Architecture Define the U-Net model architecture, which is well-suited for image segmentation tasks.
6.	Training Compile and train the U-Net model using the prepared dataset. Adjust parameters like learning rate, batch size, and epochs as needed.
7.	Evaluation Evaluate the trained model on test data to assess its performance.
8.	Results Visualize the segmentation results by comparing the model's predictions with the ground truth.

How to Run the Code
1.	Open Google Colab Go to Google Colab.
2.	Upload the Notebook Click on File -> Upload notebook. Choose cnnUnet_HepaticVessels.ipynb from the local machine to upload.
3.	Mount Google Drive Run the cell that mounts Google Drive to access the data. Make sure the dataset is stored in the appropriate directory in Google Drive.
4.	Install Necessary Libraries Execute the cells that install required libraries (nibabel and tensorflow).
5.	Configure Data Paths Modify the paths to point to the dataset in Google Drive.
6.	Run Cells Sequentially Execute each cell in the notebook in order to mount the drive, install libraries, load and preprocess data, define the model, train the model, and evaluate the results.
7.	Monitor Training and Evaluation Observe the training process and evaluate the performance of the model on the test set. Visualize the results to confirm the effectiveness of the segmentation.
___

Vessel Segmentation with U-Net
This Jupyter notebook (My_vessels_segmentation.ipynb) demonstrates the implementation of a U-Net model for segmenting vessels from medical images. The notebook is designed to run in Google Colab and utilizes Google Drive for data storage.

Setup and Installation
1.	Import Necessary Libraries The notebook begins by importing all necessary libraries for data handling, model building, and visualization.
2.	Link with Google Drive Mount Google Drive to access the dataset stored in Google Drive.
3.	Data Preparation
•	Load Input Data: Load and preprocess the dataset from Google Drive.
•	Preprocess Data: Separate and preprocess the images and labels for training and testing.
4.	Model Architecture Define the U-Net model architecture tailored for segmentation tasks.
5.	Training Compile and train the U-Net model with the prepared dataset. Adjust parameters like learning rate, batch size, and epochs as needed.
6.	Evaluation Evaluate the trained model on the test data to assess its performance.
7.	Results Visualize the segmentation results by comparing the model's predictions with the ground truth.
