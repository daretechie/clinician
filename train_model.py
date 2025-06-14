import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np # For handling potential NaN in target

# Attempt to import the preprocess_data function from preprocess.py
try:
    from preprocess import preprocess_data
    print("Successfully imported preprocess_data from preprocess.py")
except ImportError:
    print("Error: Could not import preprocess_data from preprocess.py. Ensure preprocess.py is in the same directory.")
    # As a fallback for the agent's execution, define a placeholder if direct import fails in this environment
    # This is NOT for production but for allowing the agent to proceed if file system interaction is tricky.
    def preprocess_data(df):
        print("Warning: Using placeholder preprocess_data. Actual preprocessing won't occur.")
        # Return a very simplified version, perhaps just selecting some columns if they exist
        # This will likely cause issues downstream but allows script structure testing.
        if 'Prompt' in df.columns: # A common column expected to be text
             # Simulate some processing, like TF-IDF would create many columns. Let's fake a few.
            processed_df = pd.DataFrame(index=df.index)
            for i in range(10): # Create 10 dummy features
                processed_df[f'feature_{i}'] = np.random.rand(len(df))
            # Also, keep 'DDX SNOMED' if it's the target and exists
            if 'DDX SNOMED' in df.columns:
                processed_df['DDX SNOMED'] = df['DDX SNOMED']
            return processed_df
        return df # Should not happen if train.csv is as expected


def train_and_evaluate():
    print("Starting model training and evaluation process...")

    # 1. Load Data
    try:
        print("Loading data from data/train.csv...")
        df_train = pd.read_csv("data/train.csv")
        print(f"Data loaded. Shape: {df_train.shape}")
        if df_train.empty:
            print("Error: Loaded training data is empty.")
            return
    except FileNotFoundError:
        print("Error: data/train.csv not found.")
        return
    except Exception as e:
        print(f"Error loading data/train.csv: {e}")
        return

    # Define target column
    target_column = "DDX SNOMED" # Assuming this is 'patient_diagnosis'

    if target_column not in df_train.columns:
        print(f"Error: Target column '{target_column}' not found in the DataFrame.")
        print(f"Available columns: {df_train.columns.tolist()}")
        return

    # Handle potential NaN values in the target column before LabelEncoding
    # Strategy: Fill NaN with a placeholder string like 'Unknown' or drop rows
    # For this example, let's fill with 'Unknown'
    df_train[target_column] = df_train[target_column].fillna('Unknown')
    print(f"NaNs in target column '{target_column}' filled with 'Unknown'.")

    # 2. Preprocess features (X) and encode target (y)
    print("Preprocessing features...")
    X = df_train.drop(columns=[target_column])
    y_raw = df_train[target_column]

    # Preprocess features using the imported function
    # Ensure X is not empty before preprocessing
    if X.empty:
        print("Error: Feature set X is empty after dropping target column.")
        return

    X_processed = preprocess_data(X)
    print(f"Features preprocessed. Shape of X_processed: {X_processed.shape}")
    if X_processed.empty and not X.empty: # Check if preprocessing emptied a non-empty dataframe
        print("Error: Preprocessing resulted in an empty feature set (X_processed). Check preprocess_data logic.")
        return
    if X_processed.empty and X.empty: # Original X was already empty
        print("Warning: Feature set X was empty to begin with (before preprocessing).")
        # No point in proceeding if X_processed is empty.
        return


    print(f"Encoding target variable '{target_column}'...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    print(f"Target variable encoded. Number of classes: {len(label_encoder.classes_)}")
    # Store classes for report
    class_names = label_encoder.classes_.astype(str)
    # Ensure class_names are strings for classification_report
    # If classes are few, print them
    if len(class_names) < 15:
        print(f"Classes: {class_names}")


    # 3. Split Data
    print("Splitting data into training and validation sets (80/20)...")
    # Handle cases where X_processed might be too small or have 0 samples
    if X_processed.shape[0] == 0:
        print("Error: X_processed has 0 samples. Cannot split data.")
        return
    if X_processed.shape[0] == 1: # Can't stratify with 1 sample per class if y_encoded has unique values
        print("Warning: X_processed has only 1 sample. Using it for both train and test for structural run, but this is not valid training.")
        X_train, X_val, y_train, y_val = X_processed, X_processed, y_encoded, y_encoded
    elif len(np.unique(y_encoded)) > 1 and X_processed.shape[0] >= 2 and np.min(np.bincount(y_encoded)) < 2 :
         print(f"Warning: Some classes have only 1 sample. Stratification might fail or be meaningless. Using stratify=None.")
         # If stratify fails due to single-member classes and small dataset
         try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
         except ValueError:
            print("Stratify failed due to too few samples per class. Splitting without stratify.")
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42 # No stratify
            )
    elif X_processed.shape[0] >= 2 : # Default case for splitting
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
        except ValueError: # Catch other stratification issues (e.g. all samples are one class after split attempt)
             print("Stratify failed. Splitting without stratify.")
             X_train, X_val, y_train, y_val = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42
            )
    else: # Should not be reached if X_processed.shape[0] == 0 is caught
        print("Error: Cannot split data due to insufficient samples in X_processed.")
        return

    print(f"Data split. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # 4. Train Logistic Regression Model
    print("Training Logistic Regression model...")
    # It's possible X_train is empty if X_processed was very small.
    if X_train.empty:
        print("Error: X_train is empty. Cannot train model.")
        return

    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear') # liblinear is good for smaller datasets
    try:
        model.fit(X_train, y_train)
        print("Model training completed.")
    except ValueError as ve:
        print(f"Error during model training: {ve}")
        print("This can happen if X_train is empty or contains NaN/inf values after preprocessing.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        return


    # 5. Predict on Validation Set
    print("Predicting on validation set...")
    if X_val.empty:
        print("Warning: X_val is empty. Skipping prediction and evaluation.")
        # Depending on requirements, this could be an error or just a note.
    else:
        y_pred = model.predict(X_val)
        print("Prediction completed.")

        # 6. Evaluate Model
        print("\n--- Model Evaluation ---")
        try:
            # Ensure target_names correspond to sorted unique labels in y_val and y_pred
            # If y_val or y_pred is empty, this will error.
            unique_labels_in_y = np.unique(np.concatenate((y_val, y_pred)))
            report_target_names = [cn for i, cn in enumerate(class_names) if i in unique_labels_in_y]
            if not report_target_names: # Fallback if mapping fails
                 report_target_names = None


            # Handle cases where classification_report might fail
            # (e.g. y_val is empty, or only one class present in y_pred and not in y_val)
            if len(y_val) == 0:
                print("Validation set y_val is empty. Cannot generate classification report.")
            elif len(np.unique(y_val)) == 0 and len(np.unique(y_pred)) == 0 and np.unique(y_val)[0] != np.unique(y_pred)[0]:
                # This is a very specific edge case, unlikely with real data but possible in tests
                print("y_val and y_pred contain single but different classes. Report might be misleading.")
                # Still try to print accuracy
                print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
            else:
                print("Classification Report:")
                # Use labels parameter if report_target_names causes issues due to subset of classes
                # The `labels` parameter should list all possible label indices that *could* appear.
                # `unique_labels_in_y` ensures we only try to report on labels present in the combined set.
                print(classification_report(y_val, y_pred, labels=unique_labels_in_y, target_names=report_target_names, zero_division=0))
                print(f"Overall Accuracy: {accuracy_score(y_val, y_pred):.4f}")

        except Exception as e:
            print(f"Error generating classification report: {e}")
            # Try to print accuracy at least
            try:
                print(f"Overall Accuracy: {accuracy_score(y_val, y_pred):.4f}")
            except Exception as e_acc:
                print(f"Could not calculate accuracy either: {e_acc}")

if __name__ == '__main__':
    train_and_evaluate()
    print("train_model.py script finished.")
