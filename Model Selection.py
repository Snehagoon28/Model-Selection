# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, precision_score

# Importing the dataset from the specified location
data = pd.read_csv('/Users/sneha/Documents/Machine Learning/FinalDataset1.csv')

if 'class' in data.columns:
    X = data.drop('class', axis=1)  # Features (drop the 'class' column)
    y = data['class']  # Target variable (the 'class' column)
else:
    raise ValueError("Error: 'class' column not found in the dataset.")  # Error if 'class' column is missing

# Dropping rows with missing values
data.dropna(inplace=True)

# Encoding categorical variables into numerical values using LabelEncoder
for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Redefining X and y after encoding
X = data.drop('class', axis=1) 
y = data['class'] 

# Scaling the features to standardize the data (mean=0, variance=1)
X_scaled = StandardScaler().fit_transform(X)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Defining a dictionary of models to test: Logistic Regression, KNN, and Naive Bayes
models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB()
}
precision_scores = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)  # Make predictions on the test data
    precision = precision_score(y_test, y_pred, average='weighted')  # Calculate precision score
    precision_scores[model_name] = precision  # Store precision score for comparison
    print(f"{model_name} Precision: {precision}")  # Print precision score for each model
    print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred))  # Print detailed classification report

# Extract precision scores for each model
log_reg_precision = precision_scores['Logistic Regression']
knn_precision = precision_scores['KNN']
nb_precision = precision_scores['Naive Bayes']

# Print precision scores for all models
print("Logistic Regression Precision:", log_reg_precision)
print("KNN Precision:", knn_precision)
print("Naive Bayes Precision:", nb_precision) 

# Compare the precision scores of the models and determine the best one
if log_reg_precision > knn_precision and log_reg_precision > nb_precision:
    print("The best model is Logistic Regression Model")
elif knn_precision > log_reg_precision and knn_precision > nb_precision:
    print("The best model is KNN Model.")  
    print("The best model is Naive Bayes Model.")  
