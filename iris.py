# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'C:\\Users\\Mr.Alok\\Desktop\\code alfa\\irisdata.csv'  # Update path as needed for your file location
iris_df = pd.read_csv(file_path)

# Dropping the Id column as it's not useful for prediction
iris_df_clean = iris_df.drop(columns=["Id"])  

# Encoding the target column 'Species' into numeric labels
iris_df_clean['Species'] = iris_df_clean['Species'].astype('category').cat.codes

# Splitting the data into features (X) and target (y)
X = iris_df_clean.drop(columns=["Species"])
y = iris_df_clean["Species"]

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_rep)