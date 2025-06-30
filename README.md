# Digit-Recognition-with-MNIST-Datasets-from-Kaggle
In this project I will be working with the Famous MNIST Datasets. I have downloaded the datasets from Kaggle and working in a jupyter notebook locally. 
Previously, I have used to tensorflow utilities to build a neural network. I will go with building a neural network without any fancy stuffs this time, Just Numpy and Pandas. Goal is to clearly understand the theoretical approach and apply those to build the Model. Final output will be a user interface where users can upload self captured images and get the model predictions.


Project start up Guide : 
1. Clone the project
2. Install the required dependencies : pip install -r requirements.txt
3. start the application : uvicorn main:app --reload
4. Open the postman. Create new rest endpoint as POST, Body is 'form-data', add a key 'file' and attach your image