#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# This class handles the full ML pipeline for classifying network traffic as benign or malicious.
# It includes data loading, preprocessing, model training, evaluation, and visualization.
class NetworkTrafficClassifier:
    """
    A class to run a complete machine learning pipeline for network traffic classification.

    This pipeline handles data loading, preprocessing, model training (Logistic Regression,
    Random Forest, XGBoost), evaluation on internal validation and external test sets,
    and visualization of results like confusion matrices and feature importance.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.results = {}
        self.feature_names = []
        self.output_dir = "ml_results"
        
        # Dataset names (will be set dynamically)
        self.train_dataset_name = ""
        self.test_dataset_name = ""
        
        # Create output directory name ml_results
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def get_dataset_name(self, filepath):
        """
        Extracts a clean, readable name from a dataset's file path.
        Args:
            filepath (str): The full path to the dataset file.
        Returns:
            str: The cleaned dataset name.
        """
        filename = os.path.basename(filepath)
        # Remove file extension
        name = os.path.splitext(filename)[0]
        # Clean up common suffix and make it more readable
        name = name.replace('_full', '').replace('-full', '')
        name = name.replace('_dataset', '').replace('-dataset', '')
        return name
    
    def print_section(self, title):
        """
        Prints a formatted section header for console output.
        Args:
            title (str): The title of the section.
        """
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def load_datasets(self, train_dataset_path, test_dataset_path):
        """
        It Loads the training and testing datasets from CSV files.
        This method sets the internal dataframes and prints basic info like
        the shape and label distribution of each dataset.
        Args:
            train_dataset_path (str): The file path to the training dataset.
            test_dataset_path (str): The file path to the testing dataset.
        """
        self.print_section("STEP 1: LOADING DATASETS")
        
        # Set dataset names from filenames
        self.train_dataset_name = self.get_dataset_name(train_dataset_path)
        self.test_dataset_name = self.get_dataset_name(test_dataset_path)
        
        print(f"Training dataset: {train_dataset_path}")
        print(f"Training dataset name: {self.train_dataset_name}")
        self.train_data = pd.read_csv(train_dataset_path)
        print(f"{self.train_dataset_name} dataset shape: {self.train_data.shape}")
        
        print(f"\nTesting dataset: {test_dataset_path}")
        print(f"Testing dataset name: {self.test_dataset_name}")
        self.test_data = pd.read_csv(test_dataset_path)
        print(f"{self.test_dataset_name} dataset shape: {self.test_data.shape}")
        
        # Show basic info
        print(f"\n{self.train_dataset_name} dataset label distribution:")
        print(self.train_data['Label'].value_counts())
        print(f"\n{self.test_dataset_name} dataset label distribution:")
        print(self.test_data['Label'].value_counts())
    
    def preprocess_data(self):
        """
        it Performs data cleaning and label encoding on the loaded datasets.
        this method cleans datasets by handling infinite and missing values,
        and then encodes the string labels 'Benign' and 'Malicious' into 
        integers 0 and 1, respectively. It also stores the feature names.
        """
        self.print_section("STEP 2: DATA PREPROCESSING")
        
        # Function to clean a single dataset
        def clean_dataset(df, name):
            print(f"\nProcessing {name} dataset...")
            
            # it will keep only numeric features and the Label, because models expect numeric arrays and
            # public datasets may include text metadata that would break scaling/training.
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' not in numeric_cols:
                numeric_cols.append('Label')
            
            df_clean = df[numeric_cols].copy()
            print(f"Selected {len(numeric_cols)-1} numeric features + Label column")
            
            # Handle missing and infinite values
            print(f"Missing values before cleaning: {df_clean.isnull().sum().sum()}")
            print(f"Infinite values before cleaning: {np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()}")
            
            # Replaces Inf with NaN then fill NaN per-column with median to avoid dropping rows
            # and keep distributions stable across datasets.
            df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Fill NaN with median for numeric columns (except Label)
            for col in df_clean.columns:
                if col != 'Label' and df_clean[col].dtype in ['float64', 'int64']:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
            
            print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")
            print(f"Infinite values after cleaning: {np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()}")
            
            return df_clean
        
        # Clean both datasets
        self.train_clean = clean_dataset(self.train_data, self.train_dataset_name)
        self.test_clean = clean_dataset(self.test_data, self.test_dataset_name)
        
        # Encode 'Label' as integers (Benign=0, Malicious=1). Fit encoder on TRAIN to fix mapping
        # and apply the same mapping to TEST.
        print(f"\nEncoding labels...")
        # Fit label encoder on training data
        self.label_encoder.fit(self.train_clean['Label'])
        
        self.train_clean['Label'] = self.label_encoder.transform(self.train_clean['Label'])
        self.test_clean['Label'] = self.label_encoder.transform(self.test_clean['Label'])
        
        print(f"Label encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        # Store feature names
        self.feature_names = [col for col in self.train_clean.columns if col != 'Label']
        print(f"Total features for training: {len(self.feature_names)}")
    
    def split_data(self):
        """
        Splits the training data into internal training and validation sets.
        This method creates three sets: a training set for model fitting, a
        validation set for internal performance checks, and an external test
        set for final, unbiased evaluation.
        """
        self.print_section("STEP 3: DATA SPLITTING")
        
        # Split training data into train/validation (60/40)
        X = self.train_clean[self.feature_names]
        y = self.train_clean['Label']
        
        # Stratified split preserves the class ratio in train/val (important with imbalance).
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        
        # Prepare external test set
        self.X_test_external = self.test_clean[self.feature_names]
        self.y_test_external = self.test_clean['Label']
        
        print(f"Training set size: {self.X_train.shape[0]} samples")
        print(f"Internal validation set size: {self.X_val.shape[0]} samples")
        print(f"External test set ({self.test_dataset_name}) size: {self.X_test_external.shape[0]} samples")
        
        print(f"\nTraining set label distribution:")
        print(pd.Series(self.y_train).value_counts())
    
    def scale_features(self):
        """
        Scales the feature data using StandardScaler.
        This method fits the scaler on the training data only to prevent
        data leakage, then transforms the training, validation, and test sets.
        """
        self.print_section("STEP 4: FEATURE SCALING")
        
        print("Applying StandardScaler to features...")
        
        # Fit StandardScaler on TRAIN only to prevent information leakage. Apply the same
        # transform to VALIDATION and EXTERNAL TEST.
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_external_scaled = self.scaler.transform(self.X_test_external)
        
        print("Feature scaling completed!")
        print(f"Training set mean: {self.X_train_scaled.mean():.4f}")
        print(f"Training set std: {self.X_train_scaled.std():.4f}")
    
    def handle_class_imbalance(self):
        """
        Addresses class imbalance in the training data using SMOTE.
        
        This method applies SMOTE to the scaled training features and labels
        to create a balanced dataset for model training.
        """
        self.print_section("STEP 5: HANDLING CLASS IMBALANCE")
        
        print("Original training set distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples")
        
        # Apply SMOTE only on the TRAIN split to synthetically balance classes. Do NOT apply to
        # validation/test to keep evaluation realistic.
        smote = SMOTE(random_state=42)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train_scaled, self.y_train
        )
        
        print(f"\nAfter SMOTE:")
        unique, counts = np.unique(self.y_train_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples")
    
    def train_models(self):
        """
        Trains the machine learning models.
        This method fits Logistic Regression, Random Forest, and XGBoost models
        on the balanced training data and stores them in the `self.models` dictionary.
        """
        self.print_section("STEP 6: TRAINING ML MODELS")
        
        # Use three lightweight baselines: LR (linear), RF (bagging trees), XGB (boosted trees),
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        print("Training models...")
        for name, model in models_to_train.items():
            print(f"  - Training {name}...")
            model.fit(self.X_train_balanced, self.y_train_balanced)
            self.models[name] = model
        
        print(f"Successfully trained {len(self.models)} models!")
    
    def evaluate_model(self, model_name, X_test, y_test, dataset_name):
        """
        Helper function to evaluate a trained model.
        This method calculates and prints a set of key performance metrics
        (Accuracy, Precision, Recall, F1-score, AUC) and stores them.

        Args:
            model_name (str): The name of the model to evaluate.
            X_test (np.ndarray): The feature data to evaluate on.
            y_test (pd.Series): The true labels for the evaluation data.
            dataset_name (str): The name of the dataset being evaluated (e.g., "Validation").

        Returns:
            tuple: A tuple containing the predicted labels and predicted probabilities.
        """
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Use weighted Precision/Recall/F1 to account for class imbalance. AUC computed from
        # predict_proba for the positive class (Malicious).
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        if model_name not in self.results:
            self.results[model_name] = {}
        
        self.results[model_name][dataset_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_true': y_test
        }
        
        # Print results
        print(f"\n{model_name} on {dataset_name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def evaluate_internal(self):
        """
        Evaluates all trained models on the internal validation set.
        """
        self.print_section("STEP 7: INTERNAL EVALUATION [60/40% - train/test]")
        
        internal_name = f"{self.train_dataset_name} (Validation)"
        print(f"Evaluating models on internal validation set ({internal_name})...")
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name, self.X_val_scaled, self.y_val, internal_name)
    
    def evaluate_external(self):
        """
        Evaluates all trained models on the external, unseen test set.
        """
        self.print_section("STEP 8: EXTERNAL VALIDATION[100% test]")
        
        external_name = f"{self.test_dataset_name} (Test)"
        print(f"Evaluating models on external test set ({external_name})...")
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name, self.X_test_external_scaled, self.y_test_external, external_name)
    
    def get_feature_importance(self):
        """
        Calculates and stores the feature importance for each trained model.
        This method uses a model's built-in feature importance if available (e.g.,
        for tree-based models) or uses permutation importance for linear models.
        """
        self.print_section("STEP 9: FEATURE IMPORTANCE")
        
        self.feature_importance = {}
        
        for model_name, model in self.models.items():
            print(f"\nCalculating feature importance for {model_name}...")
            
            if hasattr(model, 'feature_importances_'):
                # this tree-based models have built-in feature importance
                importance = model.feature_importances_
            else:
                # For RF/XGB, use built-in feature_importances_ (split gain); for LR use permutation
                # importance on the validation set to estimate how shuffling each feature affecting F1/AUC.
                perm_importance = permutation_importance(
                    model, self.X_val_scaled, self.y_val, n_repeats=5, random_state=42
                )
                importance = perm_importance.importances_mean
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            self.feature_importance[model_name] = importance_df
            
            print(f"Top 10 features for {model_name}:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")

    def plot_confusion_matrices(self):
        """
        Plots and saves confusion matrices for each model on each dataset.
        This method also prints the confusion matrix to the terminal with
        proper, descriptive statements.
        output: 
            Saves a PNG file named `confusion_matrices.png` to the output directory.
            Prints the confusion matrix to the console.
        """
        self.print_section("STEP 10: PLOTTING & PRINTING CONFUSION MATRICES")

        # Get dataset names for plotting and printing
        dataset_names = list(next(iter(self.results.values())).keys())
        n_models = len(self.models)
        
        # === START PLOTTING CODE ===
        fig, axes = plt.subplots(n_models, 2, figsize=(15, 4*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        # === END PLOTTING CODE ===
        
        for i, model_name in enumerate(self.models.keys()):
            for j, dataset_name in enumerate(dataset_names):
                y_true = self.results[model_name][dataset_name]['y_true']
                y_pred = self.results[model_name][dataset_name]['y_pred']
                
                cm = confusion_matrix(y_true, y_pred)
                
                # === START PLOTTING CODE ===
                # Heatmaps show absolute counts; normalizing could be added if needed for class ratio.
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['Benign', 'Malicious'],
                           yticklabels=['Benign', 'Malicious'],
                           ax=axes[i, j])
                axes[i, j].set_title(f'{model_name}\n{dataset_name}')
                axes[i, j].set_xlabel('Predicted')
                axes[i, j].set_ylabel('Actual')
                # === END PLOTTING CODE ===
                
                # === START PRINTING CODE ===
                print(f"\n--- Confusion Matrix for {model_name} on {dataset_name} ---")
                print(f"| Correctly Classified Benign: {cm[0, 0]}")
                print(f"| False Positive: (Predicted Malicious, was Benign): {cm[0, 1]}")
                print(f"| False Negative: (Predicted Benign, was Malicious): {cm[1, 0]}")
                print(f"| Correctly Classified Malicious: {cm[1, 1]}")
                print("-------------------------------------------------")
                # === END PRINTING CODE ===
                
        # === START PLOTTING CODE ===
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nConfusion matrices saved!")
        # === END PLOTTING CODE ===
        
        print("Confusion matrices also printed to terminal!")

    def plot_learning_curves(self):
        """
        Plots and saves learning curves for each trained model.
        output: Saves a PNG file named `learning_curves.png` to the
        output directory.
        """
        print("\nPlotting learning curves...")
        
        fig, axes = plt.subplots(1, len(self.models), figsize=(15, 5))
        if len(self.models) == 1:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(self.models.items()):
            # Learning curves illustrate bias/variance and stability across increasing training sizes.
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train_balanced, self.y_train_balanced,
                cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='f1_weighted'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            axes[i].plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
            axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            axes[i].plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
            axes[i].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            axes[i].set_title(f'Learning Curve - {model_name}')
            axes[i].set_xlabel('Training Set Size')
            axes[i].set_ylabel('F1 Score')
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Learning curves saved!")
    
    def plot_feature_importance(self):
        """
        Plots and saves bar charts for feature importance for each model.
        output: Saves a PNG file named `feature_importance.png` to the
        output directory.
        """
        print("\nPlotting feature importance...")
        
        fig, axes = plt.subplots(len(self.models), 1, figsize=(12, 6*len(self.models)))
        if len(self.models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(self.models.keys()):
            top_features = self.feature_importance[model_name].head(15)
            
            axes[i].barh(range(len(top_features)), top_features['Importance'])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['Feature'])
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'Top 15 Features - {model_name}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Feature importance plots saved!")
    
    def save_results_summary(self):
        """
        Saves a text and CSV summary of the pipeline's results.
        output: Creates and saves two types of files to the output
        directory: a human-readable text file with performance metrics and
        feature importance, and CSV files for each model's feature importance.
        """
        self.print_section("STEP 11: SAVING RESULTS SUMMARY")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'{self.output_dir}/results_summary_{timestamp}.txt'
        
        with open(results_file, 'w') as f:
            f.write("NETWORK TRAFFIC CLASSIFICATION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Training Dataset: {self.train_dataset_name}\n")
            f.write(f"Testing Dataset: {self.test_dataset_name}\n\n")
            
            # Model Performance Summary
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            # Get dataset names
            dataset_names = list(next(iter(self.results.values())).keys())
            
            for model_name in self.models.keys():
                f.write(f"\n{model_name}:\n")
                for dataset_name in dataset_names:
                    results = self.results[model_name][dataset_name]
                    f.write(f"  {dataset_name}:\n")
                    f.write(f"    Accuracy:  {results['accuracy']:.4f}\n")
                    f.write(f"    Precision: {results['precision']:.4f}\n")
                    f.write(f"    Recall:    {results['recall']:.4f}\n")
                    f.write(f"    F1-score:  {results['f1']:.4f}\n")
                    f.write(f"    AUC:       {results['auc']:.4f}\n")
            
            # Feature Importance Summary
            f.write(f"\n\nTOP 15 IMPORTANT FEATURES\n")
            f.write("-" * 30 + "\n")
            
            for model_name in self.models.keys():
                f.write(f"\n{model_name}:\n")
                top_15 = self.feature_importance[model_name].head(15)
                for idx, row in top_15.iterrows():
                    f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")
        
        print(f"Results summary saved to: {results_file}")
        
        # Save feature importance to CSV
        for model_name in self.models.keys():
            csv_file = f'{self.output_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.csv'
            self.feature_importance[model_name].to_csv(csv_file, index=False)
        
        print("Feature importance CSVs saved!")
    
    def display_combined_feature_analysis(self):
        """
        Analyzes and prints a combined ranking of important features across all models.
        
        This method aggregates feature importance data to identify universally important
        features and those unique to a single model. This is for display only.
        """
        self.print_section("STEP 12: COMBINED FEATURE ANALYSIS")
        
        print("Analysis of top features across all models:")
        
        # Collect top features from all models
        all_top_features = {}
        
        print(f"\nTOP 15 FEATURES FROM EACH MODEL:")
        print("=" * 60)
        
        for model_name in self.models.keys():
            print(f"\n{model_name}:")
            print("-" * 30)
            top_15 = self.feature_importance[model_name].head(15)
            for i, (_, row) in enumerate(top_15.iterrows()):
                feature = row['Feature']
                importance = row['Importance']
                print(f"  {i+1:2d}. {feature:<30}: {importance:.4f}")
                
                # Store feature with model info
                if feature not in all_top_features:
                    all_top_features[feature] = []
                all_top_features[feature].append((model_name, importance, i+1))
        
        # Create combined ranking based on average rank and frequency
        print(f"\nCOMBINED RANKING OF TOP FEATURES:")
        print("=" * 60)
        print("Features ranked by frequency and average importance across all models")
        
        combined_features = []
        for feature, model_data in all_top_features.items():
            avg_importance = np.mean([data[1] for data in model_data])
            avg_rank = np.mean([data[2] for data in model_data])
            frequency = len(model_data)  # How many models consider this important
            
            # Create a combined score (higher is better)
            # Prioritize features that appear in multiple models
            combined_score = (frequency * 10) + avg_importance - (avg_rank * 0.1)
            
            combined_features.append({
                'feature': feature,
                'avg_importance': avg_importance,
                'frequency': frequency,
                'models': [data[0] for data in model_data],
                'combined_score': combined_score
            })
        
        # Sort by combined score
        combined_features.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Display combined ranking (top 20)
        print(f"\nTop 20 most important features across all models:")
        print("-" * 60)
        for i, feat_data in enumerate(combined_features[:20]):
            feature = feat_data['feature']
            avg_imp = feat_data['avg_importance']
            freq = feat_data['frequency']
            models = feat_data['models']
            
            models_str = ", ".join([m.split()[0] for m in models])  # Abbreviate model names
            
            print(f"{i+1:2d}. {feature:<30} | Avg Imp: {avg_imp:.4f} | Models({freq}): {models_str}")
        
        # Show features that appear in all models
        universal_features = [f for f in combined_features if f['frequency'] == len(self.models)]
        if universal_features:
            print(f"\nFeatures important in ALL {len(self.models)} models:")
            print("-" * 40)
            for i, feat_data in enumerate(universal_features):
                feature = feat_data['feature']
                avg_imp = feat_data['avg_importance']
                print(f"  {i+1}. {feature:<30}: {avg_imp:.4f}")
        else:
            print(f"\nNo features appear in top 15 of all {len(self.models)} models.")
        
        # Show features unique to single models
        unique_features = [f for f in combined_features if f['frequency'] == 1]
        if unique_features:
            print(f"\nFeatures important to only ONE model:")
            print("-" * 40)
            for feat_data in unique_features[:10]:  # Show top 10 unique features
                feature = feat_data['feature']
                model = feat_data['models'][0]
                importance = feat_data['avg_importance']
                print(f"  {feature:<30}: {model} ({importance:.4f})")
        
        print(f"\nFeature Analysis Complete!")
        print(f"Total unique features in top 15 across all models: {len(all_top_features)}")
        print(f"Features appearing in multiple models: {len([f for f in combined_features if f['frequency'] > 1])}")
    
    def run_complete_pipeline(self, train_dataset_path, test_dataset_path):
        """
        Executes the entire network traffic classification pipeline.
        This method orchestrates all the steps from data loading to final
        analysis and visualization in a sequential manner.
        Args:
            train_dataset_path (str): The file path to the training dataset.
            test_dataset_path (str): The file path to the testing dataset.
        """
        print("STARTING COMPLETE ML PIPELINE")
        print("=" * 60)
        
        # Run all steps
        self.load_datasets(train_dataset_path, test_dataset_path)
        self.preprocess_data()
        self.split_data()
        self.scale_features()
        self.handle_class_imbalance()
        self.train_models()
        self.evaluate_internal()
        self.evaluate_external()
        self.get_feature_importance()
        self.plot_confusion_matrices()
        self.plot_learning_curves()
        self.plot_feature_importance()
        self.save_results_summary()
        self.display_combined_feature_analysis()
        
        self.print_section("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Training Dataset: {self.train_dataset_name}")
        print(f"Testing Dataset: {self.test_dataset_name}")
        print(f"All results saved in directory: {self.output_dir}")

# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = NetworkTrafficClassifier()
    
    # File paths (update these with your actual file paths)
    train_dataset_path = "External-Realistic-C2-Dataset.csv"
    test_dataset_path = "Synthetic-Havoc-Dataset.csv"
    
    print("Network Traffic Classification ML Pipeline")
    print("Dataset files will be automatically named in outputs based on filenames")
    print(f"Training dataset: {train_dataset_path}")
    print(f"Testing dataset: {test_dataset_path}")
    
    # Check if files exist
    if not os.path.exists(train_dataset_path):
        print(f"Error: {train_dataset_path} not found!")
        train_dataset_path = input("Enter path to training dataset: ")
    
    if not os.path.exists(test_dataset_path):
        print(f"Error: {test_dataset_path} not found!")
        test_dataset_path = input("Enter path to testing dataset: ")
    
    # Run the complete pipeline
    classifier.run_complete_pipeline(train_dataset_path, test_dataset_path)
