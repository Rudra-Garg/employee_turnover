import logging
import os
from datetime import datetime

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix)
# Machine Learning imports
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, save_model, load_model

# Import configuration
from config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmployeeTurnoverPredictor:
    """Main class for employee turnover prediction."""

    def __init__(self, config=CONFIG):
        """Initialize the predictor with configuration settings."""
        self.config = config
        self.models = {}
        self.best_models = {}
        self.scalers = {}

        # Create model save directory if it doesn't exist
        os.makedirs(config['model_saving']['save_dir'], exist_ok=True)

    def load_data(self):
        """Load and perform initial data preparation."""
        logger.info("Loading data...")
        try:
            self.df = pd.read_csv(self.config['data']['filepath'])
            self.df = self.df.rename(columns={'sales': 'department'})
            self.df["department"] = self.df["department"].replace({"support": "technical", "IT": "technical"})
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_interaction_terms(self, X):
        """Create interaction terms between numerical features."""
        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                col_name = f"interaction_{num_cols[i]}_{num_cols[j]}"
                X[col_name] = X[num_cols[i]] * X[num_cols[j]]
        return X

    def engineer_features(self, df):
        """Perform feature engineering."""
        logger.info("Engineering features...")

        # Create dummy variables
        cat_vars = ["department", "salary"]
        df_processed = df.copy()

        for var in cat_vars:
            cat_dummies = pd.get_dummies(df_processed[var], prefix=var)
            df_processed = pd.concat([df_processed, cat_dummies], axis=1)
            df_processed.drop(var, axis=1, inplace=True)

        # Create new features
        df_processed['satisfaction_to_evaluation_ratio'] = (
                df_processed['satisfaction_level'] / df_processed['last_evaluation'])
        df_processed['years_since_promotion'] = (
                df_processed['time_spend_company'] - df_processed['promotion_last_5years'] * 5)

        # Create interaction terms if configured
        if self.config['feature_engineering']['create_interaction_terms']:
            df_processed = self._create_interaction_terms(df_processed)

        # Create polynomial features if configured
        if self.config['feature_engineering']['polynomial_degree'] > 1:
            num_features = df_processed.select_dtypes(include=['float64', 'int64']).columns
            poly = PolynomialFeatures(degree=self.config['feature_engineering']['polynomial_degree'],
                                      include_bias=False)
            poly_features = poly.fit_transform(df_processed[num_features])
            poly_features_df = pd.DataFrame(poly_features[:, len(num_features):],
                                            columns=[f'poly_{i}' for i in
                                                     range(poly_features.shape[1] - len(num_features))])
            df_processed = pd.concat([df_processed, poly_features_df], axis=1)

        logger.info(f"Feature engineering completed. New shape: {df_processed.shape}")
        return df_processed

    def prepare_data(self, df):
        """Prepare data for modeling."""
        # Split features and target
        y = df['left']
        X = df.drop('left', axis=1)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.config['data']['test_size'],
                                                            random_state=self.config['data']['random_state'])

        # Scale features
        scaling_method = self.config['feature_engineering']['scaling_method']
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['feature_scaler'] = scaler

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train and tune multiple models."""
        models_to_train = {'logistic_regression': LogisticRegression(), 'random_forest': RandomForestClassifier(),
                           'xgboost': xgb.XGBClassifier(), 'lightgbm': lgb.LGBMClassifier()}

        cv = RepeatedStratifiedKFold(n_splits=self.config['cross_validation']['n_splits'],
                                     n_repeats=self.config['cross_validation']['n_repeats'],
                                     random_state=self.config['data']['random_state'])

        for name, model in models_to_train.items():
            logger.info(f"Training {name}...")
            grid_search = GridSearchCV(model, self.config['models'][name]['param_grid'],
                                       cv=cv, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            self.best_models[name] = grid_search.best_estimator_
            logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
            logger.info(f"Best score for {name}: {grid_search.best_score_:.4f}")

        # Train Neural Network
        self.train_neural_network(X_train, X_test, y_train, y_test)

    def train_neural_network(self, X_train, X_test, y_train, y_test):
        """Train a neural network model."""
        logger.info("Training Neural Network...")

        # Create validation split
        X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train,
                                                                test_size=self.config['data']['validation_split'],
                                                                random_state=self.config['data']['random_state'])

        # Build model
        model = Sequential()
        for layer in self.config['models']['neural_network']['architecture']:
            model.add(Dense(units=layer['units'], activation=layer['activation']))
            if 'dropout' in layer:
                model.add(Dropout(layer['dropout']))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=self.config['models']['neural_network']['early_stopping_patience'],
                                       restore_best_weights=True)

        # Train model
        history = model.fit(X_train_nn, y_train_nn, batch_size=self.config['models']['neural_network']['batch_size'],
                            epochs=self.config['models']['neural_network']['epochs'], validation_data=(X_val, y_val),
                            callbacks=[early_stopping])

        self.best_models['neural_network'] = model
        self.plot_nn_training_history(history)

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models."""
        results = {}
        for name, model in self.best_models.items():
            if name == 'neural_network':
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)

            results[name] = {'accuracy': accuracy_score(y_test, y_pred), 'roc_auc': roc_auc_score(y_test, y_pred),
                             'classification_report': classification_report(y_test, y_pred)}

            logger.info(f"\nResults for {name}:")
            logger.info(f"Accuracy: {results[name]['accuracy']:.4f}")
            logger.info(f"ROC AUC: {results[name]['roc_auc']:.4f}")
            logger.info("\nClassification Report:")
            logger.info(results[name]['classification_report'])

        return results

    def save_models(self):
        """Save trained models and scalers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.config['model_saving']['save_dir']

        for name, model in self.best_models.items():
            if name == 'neural_network':
                model_path = os.path.join(save_dir, f'{name}_{timestamp}')
                save_model(model, model_path)
            else:
                model_path = os.path.join(save_dir, f'{name}_{timestamp}.{self.config["model_saving"]["save_format"]}')
                joblib.dump(model, model_path)

        # Save scalers
        scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.{self.config["model_saving"]["save_format"]}')
        joblib.dump(self.scalers, scaler_path)

        logger.info("Models and scalers saved successfully.")

    def load_models(self, timestamp):
        """Load saved models and scalers."""
        save_dir = self.config['model_saving']['save_dir']

        try:
            # Load traditional ML models
            for model_name in ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']:
                model_path = os.path.join(save_dir,
                                          f'{model_name}_{timestamp}.{self.config["model_saving"]["save_format"]}')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                self.best_models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} model successfully")

            # Load Neural Network
            nn_path = os.path.join(save_dir, f'neural_network_{timestamp}')
            if not os.path.exists(nn_path):
                raise FileNotFoundError(f"Neural network model not found: {nn_path}")
            self.best_models['neural_network'] = load_model(nn_path)
            logger.info("Loaded neural network model successfully")

            # Load scalers
            scaler_path = os.path.join(save_dir, f'scaler_{timestamp}.{self.config["model_saving"]["save_format"]}')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            self.scalers = joblib.load(scaler_path)
            logger.info("Loaded scalers successfully")

            logger.info(f"All models and scalers from timestamp {timestamp} loaded successfully")

        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading models and scalers: {str(e)}")
            raise

    def plot_nn_training_history(self, history):
        """Plot neural network training history."""
        plt.figure(figsize=(12, 4))

        # Plot training & validation accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')

        # Plot training & validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, X):
        """Plot feature importance for tree-based models."""
        plt.figure(figsize=(15, 10))

        # Get feature importance from different models
        importance_data = {}

        # Random Forest
        if 'random_forest' in self.best_models:
            rf_importance = self.best_models['random_forest'].feature_importances_
            importance_data['Random Forest'] = pd.Series(rf_importance, index=X.columns)

        # XGBoost
        if 'xgboost' in self.best_models:
            xgb_importance = self.best_models['xgboost'].feature_importances_
            importance_data['XGBoost'] = pd.Series(xgb_importance, index=X.columns)

        # LightGBM
        if 'lightgbm' in self.best_models:
            lgb_importance = self.best_models['lightgbm'].feature_importances_
            importance_data['LightGBM'] = pd.Series(lgb_importance, index=X.columns)

        # Plot feature importance for each model
        for idx, (model_name, importance) in enumerate(importance_data.items()):
            plt.subplot(len(importance_data), 1, idx + 1)
            importance.sort_values(ascending=True).plot(kind='barh', title=f'{model_name} Feature Importance')
            plt.tight_layout()

        plt.show()

    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 6))

        for name, model in self.best_models.items():
            if name == 'neural_network':
                y_pred_proba = model.predict(X_test)
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models."""
        n_models = len(self.best_models)
        fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(15, 5 * ((n_models + 1) // 2)))
        axes = axes.ravel()

        for idx, (name, model) in enumerate(self.best_models.items()):
            if name == 'neural_network':
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues')
            axes[idx].set_title(f'Confusion Matrix - {name}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')

        # Remove empty subplots if odd number of models
        if n_models % 2 != 0:
            fig.delaxes(axes[-1])

        plt.tight_layout()
        plt.show()


def main():
    """Main execution function."""
    # Initialize predictor
    predictor = EmployeeTurnoverPredictor()

    try:
        # Load and prepare data
        df = predictor.load_data()

        # Engineer features
        df_processed = predictor.engineer_features(df)

        # Prepare data for modeling
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_processed)

        # Train models
        predictor.train_models(X_train, X_test, y_train, y_test)

        # Evaluate models
        results = predictor.evaluate_models(X_test, y_test)

        # Generate visualizations
        predictor.plot_feature_importance(df_processed.drop('left', axis=1))
        predictor.plot_roc_curves(X_test, y_test)
        predictor.plot_confusion_matrices(X_test, y_test)

        # Save models
        predictor.save_models()

        logger.info("Analysis completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
