import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

def preprocess_data(df_input):
    df = df_input.copy()

    numerical_features_base = ['Years of Experience']
    categorical_features_base = ['County', 'Health level', 'Nursing Competency', 'Clinical Panel']
    text_features_base = ['Prompt', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']

    # Columns that are identifiers or non-feature like, to be explicitly excluded from final feature set
    # if they were not transformed into other features.
    # 'Master_Index' is an example. If other such columns exist, add them here.
    # This list helps ensure that if they are passed through by ColumnTransformer's 'remainder',
    # they are not carried into the ML-ready feature set unless explicitly part of a transformation.
    cols_to_explicitly_drop_if_passthrough = ['Master_Index']


    actual_numerical_features = [feature for feature in numerical_features_base if feature in df.columns]
    actual_categorical_features = [feature for feature in categorical_features_base if feature in df.columns]
    actual_text_features = [feature for feature in text_features_base if feature in df.columns]

    new_categorical_feature_names = []

    if actual_text_features:
        if df.empty:
            df['combined_text'] = pd.Series(dtype='object', index=df.index)
        else:
            _text_values = df[actual_text_features].fillna('').astype(str).agg(' '.join, axis=1)
            df['combined_text'] = pd.Series(_text_values, index=df.index)
    else:
        df['combined_text'] = pd.Series([''] * len(df), index=df.index, dtype='object')

    if df.empty:
        df_for_tfidf_base_cols = []
        for col in df.columns:
            if col not in actual_text_features and \
               col not in actual_numerical_features and \
               col not in actual_categorical_features and \
               col != 'combined_text' and \
               col not in cols_to_explicitly_drop_if_passthrough: # Check here too
                df_for_tfidf_base_cols.append(col)
        df_for_tfidf_base = pd.DataFrame(columns=df_for_tfidf_base_cols, index=df.index)

    else:
        if actual_numerical_features:
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
        else:
            numerical_pipeline = 'drop'

        if actual_categorical_features:
            categorical_pipeline = Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            categorical_pipeline = 'drop'

        transformers = []
        if actual_numerical_features and numerical_pipeline != 'drop':
            transformers.append(('num', numerical_pipeline, actual_numerical_features))
        if actual_categorical_features and categorical_pipeline != 'drop':
            transformers.append(('cat', categorical_pipeline, actual_categorical_features))

        if not transformers:
            temp_cols_to_keep = []
            for col in df.columns:
                if col not in actual_text_features and \
                   col != 'combined_text' and \
                   col not in cols_to_explicitly_drop_if_passthrough: # Check here
                    temp_cols_to_keep.append(col)
            df_for_tfidf_base = df[temp_cols_to_keep].copy()
        else:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough'
            )

            df_processed_np = preprocessor.fit_transform(df)

            if actual_categorical_features and categorical_pipeline != 'drop':
                try:
                    cat_transformer_tuple = next(t for t in preprocessor.transformers_ if t[0] == 'cat')
                    onehot_encoder = cat_transformer_tuple[1].named_steps['onehot']
                    new_categorical_feature_names = list(onehot_encoder.get_feature_names_out(actual_categorical_features))
                except StopIteration:
                    print("Warning: 'cat' transformer not found.")
                except Exception as e:
                    print(f"Warning: Could not get OHE names: {e}")

            ct_output_feature_names = [] # Names of columns from CT's main transformations
            if actual_numerical_features and numerical_pipeline != 'drop':
                ct_output_feature_names.extend(actual_numerical_features)
            ct_output_feature_names.extend(new_categorical_feature_names)

            # Get names of columns that were 'passthrough'
            # These are columns in the original df that were not specified in any transformer
            df_passthrough_cols_names = [col for col in df.columns if col not in actual_numerical_features and col not in actual_categorical_features]

            # The full list of column names from ColumnTransformer's output np array
            all_ct_output_names = ct_output_feature_names + df_passthrough_cols_names

            df_processed_transformed_parts = pd.DataFrame(df_processed_np, columns=all_ct_output_names, index=df.index)

            # Construct df_for_tfidf_base: only transformed features, no other passthroughs like Master_Index
            df_for_tfidf_base_cols_final = []
            if actual_numerical_features:
                df_for_tfidf_base_cols_final.extend(actual_numerical_features)
            df_for_tfidf_base_cols_final.extend(new_categorical_feature_names)

            # We are only keeping features generated by transformers (numerical scaled, categorical OHE).
            # Any original columns that were just 'passthrough' (like Master_Index) will not be included
            # in df_for_tfidf_base unless they are part of actual_numerical_features or actual_categorical_features,
            # which they are not.
            # cols_to_explicitly_drop_if_passthrough are implicitly handled by not adding them here.

            valid_cols_for_selection = [c for c in df_for_tfidf_base_cols_final if c in df_processed_transformed_parts.columns]
            df_for_tfidf_base = df_processed_transformed_parts[valid_cols_for_selection].copy()

    if 'combined_text' in df.columns and df['combined_text'].notna().any() and df['combined_text'].str.strip().str.len().sum() > 0:
        tfidf_vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names, index=df.index)

        df_final_preprocessed = pd.concat([df_for_tfidf_base.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    else:
        df_final_preprocessed = df_for_tfidf_base.copy()
        if df_final_preprocessed.empty and df_final_preprocessed.columns.empty: # Covers df.empty and not df.empty leading to empty base
             df_final_preprocessed = pd.DataFrame(index=df.index)

    return df_final_preprocessed

if __name__ == '__main__':
    print("Preprocessing script (v7) created. Testing with dummy data...")
    data = {
        'Master_Index': ['ID1', 'ID2', 'ID3', 'ID4', 'ID5'],
        'County': ['Nairobi', 'Kiambu', 'Nairobi', 'Mombasa', 'Kiambu'],
        'Health level': ['Level 5', 'Level 4', 'Level 5', 'Level 3', 'Level 4'],
        'Years of Experience': [5, np.nan, 10, 3, 8],
        'Prompt': ["patient with fever", "child with cough", "accident victim", "routine checkup", "elderly care"],
        'Nursing Competency': ['Adult Health', 'Child Health', 'Emergency', 'General', 'Geriatrics'],
        'Clinical Panel': ['Medicine', 'Pediatrics', 'Surgery', 'Medicine', 'Medicine'],
    }
    sample_df_main_test = pd.DataFrame(data)

    try:
        print("\n--- Main Test (simulating X features, Master_Index should be dropped) ---")
        # In train_model.py, X would be df.drop(columns=['TargetColumn']), so Master_Index would be in X.
        # preprocess_data should drop Master_Index.
        x_sample = sample_df_main_test.copy()
        preprocessed_x = preprocess_data(x_sample)
        print("Preprocessed X (features only):")
        print(preprocessed_x.head(2))
        print(f"Shape of preprocessed X: {preprocessed_x.shape}")
        if 'Master_Index' in preprocessed_x.columns:
            print("ERROR: Master_Index found in preprocessed features!")
        else:
            print("OK: Master_Index NOT found in preprocessed features.")

        print("\n--- Testing with empty DataFrame ---")
        empty_df_cols = list(sample_df_main_test.columns)  # Has Master_Index
        preprocessed_empty_df = preprocess_data(pd.DataFrame(columns=empty_df_cols))
        print("Empty DataFrame after preprocessing:")
        print(preprocessed_empty_df.head(2))
        print(f"Shape of preprocessed Empty DataFrame: {preprocessed_empty_df.shape}")
        if 'Master_Index' in preprocessed_empty_df.columns:
             print("ERROR: Master_Index found in preprocessed empty df!")
        else:
            print("OK: Master_Index NOT found in preprocessed empty df (as expected, no columns).")


        print("\n--- Testing with DataFrame with only Master_Index ---")
        only_index_df = pd.DataFrame({'Master_Index': ['IDX1', 'IDX2']})
        preprocessed_only_index_df = preprocess_data(only_index_df.copy())
        print("Only Index DataFrame after preprocessing:")
        print(preprocessed_only_index_df.head(2))
        print(f"Shape of preprocessed Only Index DataFrame: {preprocessed_only_index_df.shape}")
        if 'Master_Index' in preprocessed_only_index_df.columns and not preprocessed_only_index_df.empty :
             print("ERROR: Master_Index found in preprocessed only_index_df and it's not empty!")
        elif preprocessed_only_index_df.empty :
             print("OK: Master_Index resulted in empty df (correct as it's dropped and no other features).")
        else:
            print("OK: Master_Index NOT found or df is empty.")


    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    print("\nNote: TF-IDF features (max_features=100 for these tests) can vary based on corpus.")
    print("Categorical features are one-hot encoded.")
