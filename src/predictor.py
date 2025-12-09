# src/predictor.py
"""
Enhanced Fantasy Cricket Prediction System with XGBoost Integration
Compatible with the 30-Feature System and XGBoostVarianceModel
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from src.data_loader import CricketDataLoader
from src.feature_engineer import FeatureEngineer
from src.model_training import XGBoostVarianceModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FantasyCricketPredictor:
    """Enhanced Fantasy Cricket Predictor with XGBoost 30-Feature System"""
    
    def __init__(self, models_dir: str = "outputs", data_dir: str = "data"):
        """Initialize predictor with model and data directories"""
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        self.xgboost_model = None
        self.feature_engineer = None
        self.datasets = {}
        
        logger.info(f"Initialized Fantasy Cricket Predictor with XGBoost 30-Feature System")
    
    def load_model(self, model_name: str = "xgboost_variance_model.pkl") -> bool:
        """Load trained XGBoost model"""
        try:
            self.xgboost_model = XGBoostVarianceModel()
            
            # Try to load the model
            model_path = self.models_dir / model_name
            if model_path.exists():
                if self.xgboost_model.load_model(str(model_path)):
                    logger.info(f"Successfully loaded XGBoost model: {model_name}")
                    return True
            
            # Try alternative model names and paths
            alternative_paths = [
                self.models_dir / "xgboost_variance_model.pkl",
                Path("outputs") / "xgboost_variance_model.pkl",
                Path(".") / "xgboost_variance_model.pkl"
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    if self.xgboost_model.load_model(str(alt_path)):
                        logger.info(f"Loaded model from: {alt_path}")
                        return True
            
            logger.error("No compatible XGBoost model found")
            return False
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def load_datasets(self) -> bool:
        """Load all required datasets for 30-feature engineering"""
        try:
            data_loader = CricketDataLoader(data_dir=str(self.data_dir))
            self.datasets = data_loader.load_all()
            
            # Initialize feature engineer with same parameters as training
            self.feature_engineer = FeatureEngineer(max_recent_matches=3)
            
            logger.info("Datasets loaded successfully for 30-feature prediction")
            return True
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return False
    
    def predict_fantasy_points(self, 
                             squad_team1: List[int], 
                             squad_team2: List[int],
                             venue_id: int,
                             prediction_cutoff_date: datetime) -> pd.DataFrame:
        """
        Predict fantasy points using XGBoost model with 30-feature system
        
        Args:
            squad_team1: List of player IDs for team 1
            squad_team2: List of player IDs for team 2
            venue_id: Venue identifier
            prediction_cutoff_date: Cutoff date for feature extraction
            
        Returns:
            DataFrame with predictions and player info
        """
        logger.info(f"Predicting fantasy points using XGBoost 30-feature system")
        logger.info(f"Cutoff date: {prediction_cutoff_date}")
        logger.info(f"Team 1: {len(squad_team1)} players, Team 2: {len(squad_team2)} players")
        logger.info(f"Venue: {venue_id}")
        
        # Validate inputs
        if not self.xgboost_model or not self.xgboost_model.model:
            raise ValueError("XGBoost model not loaded. Call load_model() first.")
        if not self.datasets:
            raise ValueError("Datasets not loaded. Call load_datasets() first.")
        
        all_players = squad_team1 + squad_team2
        if len(all_players) == 0:
            raise ValueError("No players provided in squads")
        
        try:
            # Step 1: Create 30-feature prediction features using FeatureEngineer
            logger.info("Creating 30-feature prediction features...")
            prediction_df = self.feature_engineer.create_prediction_features(
                player_ids=all_players,
                venue_id=venue_id,
                datasets=self.datasets,
                prediction_cutoff_date=prediction_cutoff_date
            )
            
            logger.info(f"Initial features created: {len(prediction_df)} players, {len(prediction_df.columns)} columns")
            
            # Step 2: Add team context
            logger.info("Adding team context...")
            prediction_df = self.feature_engineer.add_team_context(
                prediction_df, squad_team1, squad_team2
            )
            
            # Step 3: Make predictions using XGBoost model
            logger.info("Making predictions with XGBoost variance model...")
            prediction_results = self.xgboost_model.predict_with_variance_check(
                prediction_df, self.feature_engineer
            )
            
            logger.info(f"Generated {len(prediction_results)} XGBoost predictions")
            
            # Step 4: Analyze prediction quality
            self._analyze_prediction_quality(prediction_results['predicted_fantasy_points'])
            
            # Step 5: Enhance results with additional information
            results_df = self._enhance_prediction_results(
                prediction_results, squad_team1, squad_team2
            )
            
            # Step 6: Add team balance analysis
            self._log_team_analysis(results_df)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error during XGBoost prediction: {e}")
            raise
    
    def _analyze_prediction_quality(self, predictions: pd.Series) -> None:
        """Analyze and log prediction quality metrics"""
        logger.info(f"XGBoost Prediction Quality Analysis:")
        logger.info(f"- Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
        logger.info(f"- Prediction mean: {predictions.mean():.2f}")
        logger.info(f"- Prediction std: {predictions.std():.2f}")
        logger.info(f"- Unique predictions: {predictions.nunique()}/{len(predictions)} ({predictions.nunique()/len(predictions)*100:.1f}%)")
        
        # Quality indicators
        prediction_range = predictions.max() - predictions.min()
        logger.info(f"- Prediction range span: {prediction_range:.2f} points")
        
        # Quartile analysis
        q1, q2, q3 = predictions.quantile([0.25, 0.5, 0.75])
        logger.info(f"- Prediction quartiles: Q1={q1:.2f}, Q2={q2:.2f}, Q3={q3:.2f}")
        
        # Variance quality assessment (matching XGBoost model's variance threshold)
        variance_threshold = 10  # Same as XGBoostVarianceModel
        variance_maintained = predictions.std() >= variance_threshold
        
        logger.info(f"- Variance Quality: {'✅ GOOD' if variance_maintained else '⚠ LOW'} (threshold: {variance_threshold})")
        
        if not variance_maintained:
            logger.warning("⚠ LOW VARIATION: Predictions may be falling into averaging trap!")
        elif predictions.std() > 20.0:
            logger.info("✓ HIGH VARIATION: Excellent prediction diversity detected")
        else:
            logger.info("✓ MODERATE VARIATION: Good prediction spread")
        
        if predictions.nunique() < len(predictions) * 0.8:
            logger.warning("⚠ MANY DUPLICATES: Consider enhancing feature engineering for more variation")
        
        if prediction_range < 20:
            logger.warning("⚠ NARROW RANGE: Prediction range may be too conservative")
        else:
            logger.info("✓ GOOD RANGE: Adequate prediction spread for team selection")
    
    def _enhance_prediction_results(self, prediction_results: pd.DataFrame, 
                                  squad_team1: List[int], squad_team2: List[int]) -> pd.DataFrame:
        """Enhance prediction results with additional player information"""
        try:
            results_df = prediction_results.copy()
            
            # Ensure we have the basic prediction column
            if 'predicted_fantasy_points' not in results_df.columns:
                raise ValueError("Prediction results missing 'predicted_fantasy_points' column")
            
            # Add alias for consistency (fp_pred for backward compatibility)
            results_df['fp_pred'] = results_df['predicted_fantasy_points']
            
            # Add team information
            results_df['team_id'] = results_df['player_id'].apply(
                lambda x: 1 if x in squad_team1 else 2
            )
            
            # Add player profiles if available
            if 'profiles' in self.datasets and len(self.datasets['profiles']) > 0:
                profiles_df = self.datasets['profiles'][['player_id', 'role']].copy()
                
                # Handle name column variations
                name_cols = ['name', 'player_name', 'full_name']
                for col in name_cols:
                    if col in self.datasets['profiles'].columns:
                        profiles_df['name'] = self.datasets['profiles'][col]
                        break
                
                results_df = results_df.merge(profiles_df, on='player_id', how='left')
                
                # Fill missing values
                results_df['role'] = results_df['role'].fillna('Batter')
                if 'name' not in results_df.columns:
                    results_df['name'] = results_df['player_id'].apply(lambda x: f"Player_{x}")
            else:
                # Default values if profiles not available
                results_df['role'] = 'Batter'
                results_df['name'] = results_df['player_id'].apply(lambda x: f"Player_{x}")
            
            # Add prediction confidence based on XGBoost model characteristics
            results_df['prediction_confidence'] = self._calculate_xgboost_confidence(results_df['fp_pred'])
            
            # Add dynamic credits based on predicted performance and variance
            results_df['credits'] = self._assign_dynamic_credits(results_df['fp_pred'], results_df['role'])
            
            # Sort by predicted points (highest first)
            results_df = results_df.sort_values('fp_pred', ascending=False).reset_index(drop=True)
            
            # Add ranking information
            results_df['predicted_rank'] = results_df.index + 1
            results_df['percentile'] = (1 - (results_df.index / len(results_df))) * 100
            
            # Add variance-based insights
            results_df['variance_category'] = self._categorize_variance_predictions(results_df['fp_pred'])
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error enhancing prediction results: {e}")
            raise
    
    def _calculate_xgboost_confidence(self, predictions: pd.Series) -> pd.Series:
        """Calculate confidence scores for XGBoost predictions"""
        try:
            # XGBoost-specific confidence calculation
            # Higher variance in predictions = higher confidence in differentiation
            pred_std = predictions.std()
            
            if pred_std < 5:
                # Low variance - lower confidence
                base_confidence = 0.3
            elif pred_std < 15:
                # Moderate variance - moderate confidence
                base_confidence = 0.6
            else:
                # High variance - higher confidence
                base_confidence = 0.8
            
            # Individual prediction confidence based on distance from mean
            pred_normalized = np.abs(predictions - predictions.mean()) / (predictions.std() + 1e-8)
            individual_confidence = base_confidence + 0.2 * np.tanh(pred_normalized)
            
            # Clip to reasonable range
            confidence = individual_confidence.clip(0.2, 0.95)
            
            return confidence
            
        except Exception:
            # Return default confidence if calculation fails
            return pd.Series(0.6, index=predictions.index)
    
    def _categorize_variance_predictions(self, predictions: pd.Series) -> pd.Series:
        """Categorize predictions based on variance model insights"""
        try:
            # Calculate percentile-based categories
            p25 = predictions.quantile(0.25)
            p50 = predictions.quantile(0.50)
            p75 = predictions.quantile(0.75)
            p90 = predictions.quantile(0.90)
            
            categories = pd.Series('Medium', index=predictions.index)
            
            # High variance performers
            categories[predictions >= p90] = 'High Variance'
            categories[(predictions >= p75) & (predictions < p90)] = 'Above Average'
            categories[(predictions >= p50) & (predictions < p75)] = 'Average'
            categories[(predictions >= p25) & (predictions < p50)] = 'Below Average'
            categories[predictions < p25] = 'Low Variance'
            
            return categories
            
        except Exception:
            return pd.Series('Unknown', index=predictions.index)
    
    def _assign_dynamic_credits(self, predictions: pd.Series, roles: pd.Series) -> pd.Series:
        """Assign dynamic credits based on XGBoost predictions and variance"""
        credits = pd.Series(9.0, index=predictions.index)  # Default 9.0
        
        try:
            # Adjust credits based on predicted performance percentile
            pred_percentiles = predictions.rank(pct=True)
            
            # Use more aggressive credit scaling for XGBoost variance model
            credits += (pred_percentiles * 3.0)
            
            # Role-based adjustments (refined for XGBoost model)
            role_adjustments = {
                'All-rounder': 0.7,  # Higher adjustment for all-rounders
                'Wicketkeeper': 0.5,
                'Bowler': -0.1,       # Less penalty for bowlers
                'Batter': 0.0
            }
            
            for role, adjustment in role_adjustments.items():
                role_mask = roles.str.contains(role, case=False, na=False)
                credits[role_mask] += adjustment
            
            # Variance-based adjustments
            pred_std = predictions.std()
            if pred_std > 15:  # High variance model
                # Boost credits for top performers more aggressively
                top_20_pct = pred_percentiles >= 0.8
                credits[top_20_pct] += 0.5
            
            # Ensure credits stay within reasonable range
            credits = credits.clip(7.0, 12.5)
            
        except Exception as e:
            logger.warning(f"Error calculating dynamic credits: {e}")
            credits = pd.Series(9.0, index=predictions.index)
        
        return credits
    
    def _log_team_analysis(self, results_df: pd.DataFrame) -> None:
        """Log team balance and distribution analysis for XGBoost model"""
        try:
            # Team balance analysis
            team_stats = results_df.groupby('team_id')['fp_pred'].agg(['mean', 'std', 'count', 'min', 'max'])
            
            logger.info("XGBoost Team Analysis:")
            for team_id in [1, 2]:
                if team_id in team_stats.index:
                    stats = team_stats.loc[team_id]
                    logger.info(f"  Team {team_id}: {stats['count']} players")
                    logger.info(f"    - Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                    logger.info(f"    - Range: {stats['min']:.2f} to {stats['max']:.2f}")
            
            # Role distribution analysis
            if 'role' in results_df.columns:
                role_stats = results_df.groupby('role')['fp_pred'].agg(['mean', 'count', 'std'])
                logger.info("XGBoost Role Analysis:")
                for role, stats in role_stats.iterrows():
                    logger.info(f"  {role}: {stats['count']} players, avg={stats['mean']:.2f}, std={stats['std']:.2f}")
            
            # Variance category analysis
            if 'variance_category' in results_df.columns:
                var_stats = results_df['variance_category'].value_counts()
                logger.info("Variance Category Distribution:")
                for category, count in var_stats.items():
                    logger.info(f"  {category}: {count} players ({count/len(results_df)*100:.1f}%)")
                    
        except Exception as e:
            logger.warning(f"Error in team analysis: {e}")
    
    def analyze_predictions(self, predictions_df: pd.DataFrame) -> Dict:
        """Analyze the XGBoost predictions for insights"""
        logger.info("Analyzing XGBoost predictions...")
        
        try:
            analysis = {
                'total_players': len(predictions_df),
                'avg_prediction': float(predictions_df['fp_pred'].mean()),
                'prediction_range': (float(predictions_df['fp_pred'].min()), 
                                   float(predictions_df['fp_pred'].max())),
                'prediction_std': float(predictions_df['fp_pred'].std()),
                'unique_predictions': predictions_df['fp_pred'].nunique(),
                'prediction_variety': predictions_df['fp_pred'].nunique() / len(predictions_df),
                'model_type': 'XGBoost Variance Model with 30-Feature System'
            }
            
            # Variance quality assessment
            pred_std = predictions_df['fp_pred'].std()
            variance_threshold = 10
            analysis['variance_maintained'] = pred_std >= variance_threshold
            analysis['variance_quality'] = (
                'Excellent' if pred_std > 20 else
                'Good' if pred_std > 15 else
                'Moderate' if pred_std > 10 else
                'Poor'
            )
            
            # Range analysis
            pred_range = predictions_df['fp_pred'].max() - predictions_df['fp_pred'].min()
            analysis['prediction_range_span'] = float(pred_range)
            analysis['range_quality'] = 'Good' if pred_range > 40 else 'Moderate' if pred_range > 20 else 'Poor'
            
            # Team analysis
            if 'team_id' in predictions_df.columns:
                team_stats = predictions_df.groupby('team_id')['fp_pred'].agg(['mean', 'std', 'count'])
                analysis['team_breakdown'] = {
                    int(team_id): {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'count': int(stats['count'])
                    }
                    for team_id, stats in team_stats.iterrows()
                }
            
            # Role analysis
            if 'role' in predictions_df.columns:
                role_stats = predictions_df.groupby('role')['fp_pred'].agg(['mean', 'std', 'count'])
                analysis['role_breakdown'] = {
                    role: {
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'count': int(stats['count'])
                    }
                    for role, stats in role_stats.iterrows()
                }
            
            # XGBoost-specific metrics
            analysis['xgboost_metrics'] = {
                'avoiding_average_trap': analysis['variance_maintained'],
                'prediction_diversity_score': analysis['prediction_variety'],
                'variance_score': pred_std / 25.0,  # Normalized variance score
                'feature_utilization': '30-feature context system'
            }
            
            # Performance tier analysis with XGBoost variance focus
            high_var = (predictions_df['fp_pred'] >= predictions_df['fp_pred'].quantile(0.85)).sum()
            above_avg = ((predictions_df['fp_pred'] >= predictions_df['fp_pred'].quantile(0.6)) & 
                        (predictions_df['fp_pred'] < predictions_df['fp_pred'].quantile(0.85))).sum()
            avg_tier = ((predictions_df['fp_pred'] >= predictions_df['fp_pred'].quantile(0.4)) & 
                       (predictions_df['fp_pred'] < predictions_df['fp_pred'].quantile(0.6))).sum()
            below_avg = ((predictions_df['fp_pred'] >= predictions_df['fp_pred'].quantile(0.15)) & 
                        (predictions_df['fp_pred'] < predictions_df['fp_pred'].quantile(0.4))).sum()
            low_var = (predictions_df['fp_pred'] < predictions_df['fp_pred'].quantile(0.15)).sum()
            
            analysis['performance_tiers'] = {
                'high_variance': int(high_var),
                'above_average': int(above_avg),
                'average': int(avg_tier),
                'below_average': int(below_avg),
                'low_variance': int(low_var)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing predictions: {e}")
            return {'error': str(e)}
    
    def print_predictions(self, predictions_df: pd.DataFrame, show_top: int = 15):
        """Print formatted prediction results with XGBoost analysis"""
        print("\n" + "="*100)
        print("XGBOOST VARIANCE FANTASY CRICKET PREDICTIONS (30-Feature System)")
        print("="*100)
        
        print(f"\nTOP {show_top} PREDICTED PERFORMERS:")
        print(f"{'Rank':<4} | {'Player ID':<10} | {'Name':<20} | {'Team':<4} | {'Role':<12} | {'Points':<8} | {'Credits':<7} | {'Conf':<5} | {'Category':<12}")
        print("-" * 100)
        
        top_players = predictions_df.head(show_top)
        for idx, (_, player) in enumerate(top_players.iterrows(), 1):
            player_id = player['player_id']
            name = str(player.get('name', f"Player_{player_id}"))[:20]
            team = f"T{player['team_id']}"
            role = player.get('role', 'Unknown')[:12]
            points = player['fp_pred']
            credits = player.get('credits', 9.0)
            confidence = player.get('prediction_confidence', 0.6)
            category = str(player.get('variance_category', 'Unknown'))[:12]
            
            print(f"{idx:<4} | {player_id:<10} | {name:<20} | {team:<4} | {role:<12} | {points:<8.2f} | {credits:<7.1f} | {confidence:<5.2f} | {category:<12}")
        
        # XGBoost analysis
        analysis = self.analyze_predictions(predictions_df)
        
        print(f"\nXGBOOST VARIANCE MODEL ANALYSIS:")
        print(f"- Model Type: {analysis.get('model_type', 'XGBoost Variance Model')}")
        print(f"- Total players: {analysis.get('total_players', len(predictions_df))}")
        print(f"- Average predicted points: {analysis.get('avg_prediction', 0):.2f}")
        
        pred_range = analysis.get('prediction_range', (0, 0))
        print(f"- Prediction range: {pred_range[0]:.2f} to {pred_range[1]:.2f}")
        print(f"- Range span: {analysis.get('prediction_range_span', 0):.2f} points ({analysis.get('range_quality', 'Unknown')})")
        print(f"- Standard deviation: {analysis.get('prediction_std', 0):.2f}")
        print(f"- Variance quality: {analysis.get('variance_quality', 'Unknown')}")
        print(f"- Avoiding average trap: {'✅ YES' if analysis.get('variance_maintained', False) else '❌ NO'}")
        print(f"- Unique predictions: {analysis.get('unique_predictions', 0)}/{analysis.get('total_players', 0)} ({analysis.get('prediction_variety', 0)*100:.1f}%)")
        
        # XGBoost specific metrics
        xgb_metrics = analysis.get('xgboost_metrics', {})
        print(f"- Prediction diversity score: {xgb_metrics.get('prediction_diversity_score', 0):.3f}")
        print(f"- Variance score: {xgb_metrics.get('variance_score', 0):.3f}/1.0")
        
        # Team balance
        if 'team_breakdown' in analysis:
            team_means = {team_id: stats['mean'] for team_id, stats in analysis['team_breakdown'].items()}
            team_stds = {team_id: stats['std'] for team_id, stats in analysis['team_breakdown'].items()}
            t1_avg, t1_std = team_means.get(1, 0), team_stds.get(1, 0)
            t2_avg, t2_std = team_means.get(2, 0), team_stds.get(2, 0)
            print(f"- Team Balance: T1={t1_avg:.1f}±{t1_std:.1f}, T2={t2_avg:.1f}±{t2_std:.1f}")
        
        # Performance tiers
        if 'performance_tiers' in analysis:
            tiers = analysis['performance_tiers']
            print(f"- Performance Tiers: High={tiers['high_variance']}, Above={tiers['above_average']}, Avg={tiers['average']}, Below={tiers['below_average']}, Low={tiers['low_variance']}")
        
        print("="*100)
    
    def save_predictions(self, predictions_df: pd.DataFrame, filename: str = "xgboost_predictions.csv"):
        """Save XGBoost predictions to CSV file"""
        try:
            # Add metadata columns for analysis
            save_df = predictions_df.copy()
            save_df['model_type'] = 'XGBoost_Variance_30_Features'
            save_df['prediction_timestamp'] = datetime.now().isoformat()
            
            save_df.to_csv(filename, index=False)
            logger.info(f"XGBoost predictions saved to {filename}")
            
            # Also save detailed analysis
            analysis_filename = filename.replace('.csv', '_analysis.txt')
            analysis = self.analyze_predictions(predictions_df)
            
            with open(analysis_filename, 'w') as f:
                f.write("XGBoost Variance Fantasy Cricket Prediction Analysis\n")
                f.write("=" * 60 + "\n\n")
                for key, value in analysis.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info(f"Prediction analysis saved to {analysis_filename}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained XGBoost model"""
        try:
            if self.xgboost_model is None:
                return None
                
            return self.xgboost_model.get_feature_importance()
            
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return None
    
    def get_prediction_distribution_analysis(self, predictions_df: pd.DataFrame) -> Dict:
        """Get detailed prediction distribution analysis"""
        try:
            if self.xgboost_model is None:
                return {}
                
            predictions = predictions_df['fp_pred'].values
            return self.xgboost_model.analyze_prediction_distribution(predictions)
            
        except Exception as e:
            logger.warning(f"Could not analyze prediction distribution: {e}")
            return {}


# Updated function for direct use in main.py
def predict_fantasy_points(squad_team1: List[int], 
                          squad_team2: List[int],
                          venue_id: int,
                          prediction_cutoff_date: datetime,
                          models_dir: str = "outputs",
                          data_dir: str = "data") -> pd.DataFrame:
    """
    Simple function to get XGBoost fantasy point predictions with 30-feature system
    
    Returns:
        DataFrame with player_id, fp_pred, team_id, role, name, credits, prediction_confidence, variance_category columns
    """
    predictor = FantasyCricketPredictor(models_dir=models_dir, data_dir=data_dir)
    
    if not predictor.load_model():
        raise RuntimeError("Failed to load XGBoost model")
    
    if not predictor.load_datasets():
        raise RuntimeError("Failed to load datasets")
    
    return predictor.predict_fantasy_points(
        squad_team1=squad_team1,
        squad_team2=squad_team2,
        venue_id=venue_id,
        prediction_cutoff_date=prediction_cutoff_date
    )