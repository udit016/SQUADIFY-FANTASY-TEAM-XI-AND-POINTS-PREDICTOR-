# src/xgboost_variance_model.py
"""
XGBoost Model Training for Fantasy Cricket - Variance Focused
Fixed compatibility issues with latest XGBoost versions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from typing import Dict, Tuple
import pickle
import os

logger = logging.getLogger(__name__)


class XGBoostVarianceModel:
    """
    XGBoost model specifically trained to predict fantasy score variance
    Compatible with latest XGBoost versions
    """
    
    def __init__(self):
        """Initialize the variance-focused XGBoost model"""
        self.model = None
        self.feature_columns = []
        self.training_metrics = {}
        self.variance_threshold = 10  # Minimum prediction std to avoid averaging
        
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        logger.info("Initialized XGBoost Variance Model for fantasy cricket prediction")
    
    def train_variance_model(self, training_df: pd.DataFrame, 
                           feature_engineer, target_col: str = 'fantasy_points') -> Dict:
        """
        Train XGBoost model with variance-focused approach
        Fixed for latest XGBoost compatibility
        """
        logger.info("Training XGBoost model with variance-focused approach...")
        
        try:
            # Prepare training data
            df = feature_engineer.prepare_features_for_training(training_df.copy())
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found")
            
            # Get feature columns
            self.feature_columns = feature_engineer.get_feature_columns()
            logger.info(f"Training with {len(self.feature_columns)} context-based features")
            
            # Prepare feature matrix and target
            X = df[self.feature_columns].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            # Remove extreme outliers
            outlier_mask = (y >= -5) & (y <= 200)
            X = X[outlier_mask]
            y = y[outlier_mask]
            
            logger.info(f"Training data shape: {X.shape}")
            logger.info(f"Target distribution - Mean: {y.mean():.1f}, Std: {y.std():.1f}, Range: [{y.min():.1f}, {y.max():.1f}]")
            
            # Check if target has sufficient variance
            if y.std() < 5:
                logger.warning("Target variable has very low variance - model may struggle with variance prediction")
            
            # Configure XGBoost for variance prediction - FIXED PARAMETERS
            xgb_params = {
                'objective': 'reg:squarederror',
                
                # Tree structure - optimized for variance capture
                'max_depth': 8,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                
                # Learning parameters
                'learning_rate': 0.05,
                'n_estimators': 500,
                
                # Regularization to prevent averaging
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'gamma': 0.1,
                
                # Other parameters
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0,
                
                # Enable early stopping
                'enable_categorical': False,
                'validate_parameters': True
            }
            
            # Time-series aware cross-validation
            logger.info("Performing time-series cross-validation...")
            tscv = TimeSeriesSplit(n_splits=5)
            
            cv_scores = []
            cv_variance_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"Training fold {fold + 1}/5...")
                
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train fold model - FIXED FIT CALL
                fold_model = xgb.XGBRegressor(**xgb_params)
                
                try:
                    # Try new XGBoost API first
                    fold_model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except TypeError:
                    # Fallback for older XGBoost versions
                    logger.info("Using legacy XGBoost API for early stopping")
                    fold_model.fit(X_train_fold, y_train_fold)
                
                # Evaluate fold
                y_pred_fold = fold_model.predict(X_val_fold)
                fold_rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                fold_variance_score = self._evaluate_variance_prediction(y_val_fold, y_pred_fold)
                
                cv_scores.append(fold_rmse)
                cv_variance_scores.append(fold_variance_score)
                
                logger.info(f"Fold {fold + 1} - RMSE: {fold_rmse:.2f}, Variance Score: {fold_variance_score:.3f}")
            
            # Log cross-validation results
            mean_cv_rmse = np.mean(cv_scores)
            mean_variance_score = np.mean(cv_variance_scores)
            
            logger.info(f"Cross-validation results:")
            logger.info(f"  Mean RMSE: {mean_cv_rmse:.2f} (+/- {np.std(cv_scores)*2:.2f})")
            logger.info(f"  Mean Variance Score: {mean_variance_score:.3f} (+/- {np.std(cv_variance_scores)*2:.3f})")
            
            # Train final model on all data - FIXED FIT CALL
            logger.info("Training final model on full dataset...")
            
            self.model = xgb.XGBRegressor(**xgb_params)
            
            try:
                # Try new XGBoost API first
                self.model.fit(
                    X, y,
                    eval_set=[(X, y)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            except TypeError:
                # Fallback for older XGBoost versions
                logger.info("Using legacy XGBoost API - training without early stopping")
                self.model.fit(X, y)
            
            # Evaluate final model variance prediction capability
            final_predictions = self.model.predict(X)
            final_variance_score = self._evaluate_variance_prediction(y, final_predictions)
            final_rmse = np.sqrt(mean_squared_error(y, final_predictions))
            final_r2 = r2_score(y, final_predictions)
            
            # Check if model avoids averaging trap
            pred_std = np.std(final_predictions)
            avoiding_avg_trap = pred_std >= self.variance_threshold
            
            logger.info("=== FINAL MODEL EVALUATION ===")
            logger.info(f"Training RMSE: {final_rmse:.2f}")
            logger.info(f"Training R²: {final_r2:.3f}")
            logger.info(f"Variance Score: {final_variance_score:.3f}")
            logger.info(f"Prediction Std: {pred_std:.1f} (threshold: {self.variance_threshold})")
            logger.info(f"Avoiding Average Trap: {'✅ YES' if avoiding_avg_trap else '❌ NO'}")
            
            if not avoiding_avg_trap:
                logger.warning("Model may be falling into averaging trap - predictions have low variance")
            
            # Store training metrics
            self.training_metrics = {
                'final_rmse': float(final_rmse),
                'final_r2': float(final_r2),
                'final_variance_score': float(final_variance_score),
                'cv_mean_rmse': float(mean_cv_rmse),
                'cv_mean_variance_score': float(mean_variance_score),
                'prediction_std': float(pred_std),
                'avoiding_average_trap': avoiding_avg_trap,
                'training_samples': len(X),
                'feature_count': len(self.feature_columns),
                'target_std': float(y.std()),
                'best_iteration': getattr(self.model, 'best_iteration', xgb_params['n_estimators'])
            }
            
            # Save model and features
            model_path = 'outputs/xgboost_variance_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            features_path = 'outputs/xgboost_feature_columns.pkl'
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Features saved to {features_path}")
            
            return {
                'model': self.model,
                'metrics': self.training_metrics,
                'feature_columns': self.feature_columns,
                'avoiding_average_trap': avoiding_avg_trap,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost variance model: {e}")
            raise
    
    def predict_with_variance_check(self, prediction_df: pd.DataFrame, 
                                  feature_engineer) -> pd.DataFrame:
        """Make predictions and verify variance is maintained"""
        logger.info("Making predictions with variance check...")
        
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Prepare prediction data
            df = feature_engineer.prepare_features_for_prediction(prediction_df.copy())
            
            if 'player_id' not in df.columns:
                raise ValueError("player_id required for predictions")
            
            player_ids = df['player_id'].copy()
            
            # Prepare features
            X_pred = df[self.feature_columns].copy()
            X_pred = X_pred.fillna(0)
            X_pred = X_pred.replace([np.inf, -np.inf], 0)
            
            # Make predictions
            predictions = self.model.predict(X_pred)
            
            # Clip to reasonable bounds
            predictions = np.clip(predictions, -2, 200)
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'player_id': player_ids,
                'predicted_fantasy_points': predictions
            })
            
            # Check prediction variance
            pred_std = np.std(predictions)
            pred_range = np.max(predictions) - np.min(predictions)
            variance_maintained = pred_std >= self.variance_threshold
            
            logger.info("=== PREDICTION VARIANCE CHECK ===")
            logger.info(f"Predictions - Mean: {np.mean(predictions):.1f}, Std: {pred_std:.1f}")
            logger.info(f"Range: [{np.min(predictions):.1f}, {np.max(predictions):.1f}]")
            logger.info(f"Variance Maintained: {'✅ YES' if variance_maintained else '❌ NO'}")
            
            if not variance_maintained:
                logger.warning("Predictions show low variance - model may be averaging")
            
            # Sort by predicted score (descending)
            results_df = results_df.sort_values('predicted_fantasy_points', ascending=False)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def _evaluate_variance_prediction(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Custom evaluation metric for variance prediction quality
        
        Returns:
            Score between 0 and 1, where 1 is perfect variance prediction
        """
        try:
            # Calculate variance ratio
            true_variance = np.var(y_true)
            pred_variance = np.var(y_pred)
            
            if true_variance == 0:
                return 0.0
            
            variance_ratio = pred_variance / true_variance
            
            # Score based on how close variance ratio is to 1.0
            # Perfect score (1.0) when ratio = 1.0
            # Decreases as ratio moves away from 1.0
            variance_score = 1.0 / (1.0 + abs(variance_ratio - 1.0))
            
            return variance_score
            
        except Exception:
            return 0.0
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from trained XGBoost model"""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Get feature importance - FIXED FOR COMPATIBILITY
            try:
                # Try new XGBoost API
                importance_scores = self.model.feature_importances_
            except AttributeError:
                # Fallback for older versions
                importance_dict = self.model.get_booster().get_score(importance_type='gain')
                importance_scores = [importance_dict.get(f, 0) for f in self.feature_columns]
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance_scores,
                'importance_type': 'gain'
            }).sort_values('importance', ascending=False).head(top_n)
            
            logger.info(f"Top {min(10, len(importance_df))} most important features:")
            for i, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def load_model(self, model_path: str = 'outputs/xgboost_variance_model.pkl') -> bool:
        """Load trained XGBoost model"""
        try:
            logger.info(f"Loading XGBoost model from {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load feature columns
            features_path = 'outputs/xgboost_feature_columns.pkl'
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                
                logger.info(f"XGBoost variance model loaded successfully")
                logger.info(f"Features: {len(self.feature_columns)}")
                return True
            else:
                logger.error("Feature columns file not found")
                return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def analyze_prediction_distribution(self, predictions: np.ndarray) -> Dict:
        """Analyze prediction distribution to check variance quality"""
        try:
            analysis = {
                'total_predictions': len(predictions),
                'mean': float(np.mean(predictions)),
                'std': float(np.std(predictions)),
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'range': float(np.max(predictions) - np.min(predictions)),
                'percentiles': {
                    'p10': float(np.percentile(predictions, 10)),
                    'p25': float(np.percentile(predictions, 25)),
                    'p50': float(np.percentile(predictions, 50)),
                    'p75': float(np.percentile(predictions, 75)),
                    'p90': float(np.percentile(predictions, 90))
                },
                'distribution': {
                    'low_scores': int(np.sum(predictions < 20)),
                    'medium_scores': int(np.sum((predictions >= 20) & (predictions < 50))),
                    'high_scores': int(np.sum(predictions >= 50)),
                    'very_high_scores': int(np.sum(predictions >= 70))
                },
                'variance_quality': {
                    'sufficient_spread': predictions.std() >= self.variance_threshold,
                    'coefficient_of_variation': float(predictions.std() / predictions.mean()) if predictions.mean() > 0 else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing prediction distribution: {e}")
            return {}
    
    def check_xgboost_version(self):
        """Check XGBoost version for debugging"""
        import xgboost as xgb
        logger.info(f"XGBoost version: {xgb.__version__}")
        return xgb.__version__