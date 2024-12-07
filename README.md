Multi-Model Classification Comparison
This project compares the performance of three machine learning models—Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes—on a classification dataset. The goal is to determine the best-performing model based on precision scores and detailed classification metrics.

Features:
Dataset Preparation:
The dataset is loaded and validated to ensure it contains a class column as the target variable.
Rows with missing values are dropped to ensure data integrity.
Categorical variables are encoded into numerical format using LabelEncoder.

Feature Scaling:
The feature set is standardized using StandardScaler to normalize the data, ensuring compatibility with distance-based algorithms like KNN.

Train-Test Split:
The dataset is split into training (80%) and testing (20%) subsets to evaluate model performance on unseen data.

Model Evaluation:
Three models are trained and evaluated:
Logistic Regression
K-Nearest Neighbors (KNN)
Naive Bayes
Precision scores are calculated using the weighted average approach to handle class imbalances.
Detailed classification reports are generated for each model to provide metrics such as precision, recall, F1-score, and support.

Model Comparison:
Precision scores for all models are compared to identify the best-performing algorithm.
The top-performing model is highlighted based on precision.
