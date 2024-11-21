"""Configuration settings for the employee turnover prediction model."""

CONFIG = {
    # Data settings
    'data': {
        'filepath': 'HR_comma_sep.csv',
        'test_size': 0.3,
        'random_state': 42,
        'validation_split': 0.2
    },

    # Feature engineering settings
    'feature_engineering': {
        'create_interaction_terms': False,
        'polynomial_degree': 1,
        'scaling_method': 'standard'  # Options: 'standard', 'minmax', 'robust'
    },

    # Model parameters
    'models': {
        'logistic_regression': {
            'param_grid': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'random_forest': {
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'xgboost': {
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'lightgbm': {
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'num_leaves': [31, 62, 93],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        },
        'neural_network': {
            'architecture': [
                {'units': 64, 'activation': 'relu', 'dropout': 0.3},
                {'units': 32, 'activation': 'relu', 'dropout': 0.2},
                {'units': 16, 'activation': 'relu', 'dropout': 0.1},
                {'units': 1, 'activation': 'sigmoid'}
            ],
            'batch_size': 32,
            'epochs': 50,
            'early_stopping_patience': 5
        }
    },

    # Cross-validation settings
    'cross_validation': {
        'n_splits': 5,
        'n_repeats': 3
    },

    # Model saving settings
    'model_saving': {
        'save_dir': 'saved_models',
        'save_format': 'joblib'  # Options: 'joblib', 'pickle'
    }
}