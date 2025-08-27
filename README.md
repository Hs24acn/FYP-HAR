# Human Activity Recognition Using Smartphone Sensor Data

## Project Overview
This project implements machine learning models to classify human activities using smartphone sensor data. The system can recognize six different activities: LAYING, SITTING, STANDING, WALKING, WALKING_DOWNSTAIRS, and WALKING_UPSTAIRS.

## Dataset
- **Training Data**: 7,352 samples with 563 features
- **Testing Data**: 2,947 samples with 563 features
- **Features**: Smartphone sensor data including:
  - Body acceleration (mean, std, mad, max, min)
  - Body gyroscope data
  - Frequency domain features
  - Angle calculations between various sensor readings

## Models Implemented

### 1. LSTM Neural Network
- **Architecture**: Sequential model with LSTM layer (64 units) + Dropout (0.5) + Dense output layer (6 units)
- **Activation**: Softmax for multi-class classification
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Performance**: Achieves high accuracy through deep learning approach

### 2. Random Forest
- **Configuration**: 100 estimators with random state 42
- **Performance**: 92.47% validation accuracy
- **Strengths**: Good overall performance, handles non-linear relationships

### 3. Support Vector Machine (SVM)
- **Configuration**: RBF kernel, C=1, gamma='scale'
- **Performance**: 95.11% validation accuracy (best performing model)
- **Strengths**: Excellent classification performance across all activity classes

## Data Preprocessing
1. **Feature Separation**: Split features (X) and target labels (y)
2. **Label Encoding**: Convert categorical activity labels to numeric values (0-5)
3. **Normalization**: Apply MinMaxScaler to scale features between 0 and 1
4. **Data Reshaping**: Reshape data for LSTM input (samples, time_steps, features)

## Activity Classes
| Label | Activity |
|-------|----------|
| 0 | LAYING |
| 1 | SITTING |
| 2 | STANDING |
| 3 | WALKING |
| 4 | WALKING_DOWNSTAIRS |
| 5 | WALKING_UPSTAIRS |

## Dependencies
```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow/keras
```

## Usage
1. Ensure the dataset files (`train.csv`, `test.csv`) are in the `dataset/` folder
2. Run the Jupyter notebook `Code.ipynb`
3. The notebook will automatically:
   - Load and preprocess the data
   - Train all three models
   - Display performance metrics and visualizations

## Results Summary
- **LSTM**: Deep learning approach with sequential data processing
- **Random Forest**: 92.47% accuracy, good for baseline comparison
- **SVM**: 95.11% accuracy, best overall performance
- **Data Quality**: No missing values, well-balanced class distribution

## Project Structure
```
├── Code.ipynb          # Main implementation notebook
├── dataset/            # Data folder
│   ├── train.csv      # Training dataset
│   └── test.csv       # Testing dataset
└── README.md          # This file
```


## Author
Hamza Shaukat - Final Year Project (FYP3)
