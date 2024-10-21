
# Heart Disease Prediction Using Neural Networks

## Overview
This project aims to predict heart disease using a binary classification model based on various health indicators. The project involves data preprocessing, building an artificial neural network (ANN) using Keras, and tuning the hyperparameters using **Keras Tuner**. The goal is to achieve high accuracy in predicting whether a person has heart disease based on health metrics such as BMI, smoking habits, blood pressure, and more.

## Dataset
The dataset used for this project is `heart_disease_health_indicators.csv`, which contains health information for various individuals. The target variable is `HeartDiseaseorAttack`, where:
- `0`: No heart disease
- `1`: Has heart disease or has had an attack

### Features
- **HighBP**: Binary indicator for high blood pressure (0 = No, 1 = Yes)
- **HighChol**: Binary indicator for high cholesterol (0 = No, 1 = Yes)
- **Smoker**: Binary indicator for smoking status (0 = No, 1 = Yes)
- **BMI**: Body Mass Index (numerical)
- **PhysicalActivity**: Binary indicator for physical activity (0 = No, 1 = Yes)
- **... (other features)**

### Target
- **HeartDiseaseorAttack**: Binary classification label (0 or 1)

## Project Steps

### 1. Data Preprocessing
- Load the dataset and handle any missing or NaN values. In this case, forward filling (`ffill`) was used to deal with missing data.
- Standardize the features using `StandardScaler` to normalize the input data for better performance of the neural network.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Train-Test Split
- The data is split into training (80%) and test (20%) sets to evaluate the model's performance.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### 3. Neural Network Architecture
The neural network is built using the **Keras Sequential API** with the following structure:
- Input Layer
- Two hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
- Output layer with 1 neuron and a sigmoid activation for binary classification.

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### 4. Compilation and Training
- The model is compiled with the **Adam** optimizer and a learning rate of 0.0001. The loss function is `binary_crossentropy`, suitable for binary classification, and the performance is measured using accuracy.

```python
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```

- The model is trained for 50 epochs using a validation split of 20%.

```python
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)
```

### 5. Model Evaluation
The trained model is evaluated using the test data. Key performance metrics include:
- **Confusion Matrix**: Shows the true positives, true negatives, false positives, and false negatives.
- **Classification Report**: Includes precision, recall, F1-score, and support for each class.
- **ROC-AUC Score**: Indicates the quality of classification.

```python
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
```

### 6. Hyperparameter Tuning with Keras Tuner
To further improve the model's performance, **Keras Tuner** is used to find the best hyperparameters, such as:
- Number of units in the hidden layers
- Number of hidden layers
- Learning rate

Keras Tuner explores different configurations and selects the best performing model based on validation accuracy.

```python
import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(X_train.shape[1],)))
    
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32), activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(build_model, objective='val_accuracy', max_trials=5, executions_per_trial=1, directory='my_dir', project_name='heart_disease_tuning')
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, batch_size=32)
```

### 7. Results
The best hyperparameters found by the tuner are printed, and the best model is retrained and evaluated on the test set. Metrics like the ROC-AUC score, confusion matrix, and classification report are used to assess the performance.

## Requirements
- Python 3.x
- TensorFlow / Keras
- Keras Tuner
- Pandas
- Scikit-learn

To install the required libraries, run:

```bash
pip install tensorflow keras pandas scikit-learn keras-tuner
```

## Project Structure
- `heart_disease_health_indicators.csv`: The dataset file
- `main.ipynb`: The Jupyter notebook containing the full code (data preprocessing, model building, hyperparameter tuning, evaluation)
- `README.md`: This file

## Deliverables
- **Jupyter Notebook**: Includes all the steps (data preprocessing, model building, evaluation, hyperparameter tuning).
- **Graphs/Plots**: Showing loss during training, ROC-AUC curve, confusion matrix.
- **GitHub Repository**: All deliverables (including the code, dataset, and results) will be uploaded to a GitHub repository.

---

This **ReadMe** provides clear instructions about your project, helping others to understand and reproduce your results.
