##### **_All raw data was taken from The UC Irvine Machine Learning Respository https://archive.ics.uci.edu/ml/_**

## **Background** 
This is a simple neural network with one hidden layer that takes an initial neural network and trains it on normalized data.
The user can change the perceptrons in the hidden layer as well as the learning rate and epochs for optimality. 
Once the neural network is trained, it can be tested on new data.
The results include accuracy, precision, F1, and recall.

## **Description of Datasets**

### **Breast Cancer**
The neural network was used to analyze breast cancer data to determine if a tumor is benign or malignant. 
The attributes measured included the turmor's: radius, texture, perimeter, area, smoothness, compactness, concavity,
symmetry, and fractal dimension. Classified tumors with 96% accuracy.

### **Parkinson's Disease**
The neural network was also used to analyze Parkinson's data to determine whether or not someone has Parkinson's by analyzing
biomedical voice measurements. The attributes measured included: Average maximum, and minumum vocal fundamental frequencies,
several measures of variation in fundamental frequency, several measures of variation in amplitude, ratio of nose to tonal
components, nonlinear dynamical complexity measures, and signal fractal scaling exponents.

## **How to run the program**
Once the folder is open in terminal:
```
python3 main.py
```

The user will then be prompted to either train or test the neural network.
