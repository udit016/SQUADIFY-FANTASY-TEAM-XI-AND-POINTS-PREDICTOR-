# src/feature_engineer.py
"""
Enhanced Feature Engineering for Fantasy Cricket Prediction
30 context-based features to break averaging traps:
- 10 Recent Performance (Raw values only)
- 5 Player Archetype (Current year priority)
- 5 Ball-by-Ball Context
- 4 Opposition Context
- 4 Venue Context
- 2 Match Context
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates 30-feature system for fantasy cricket prediction
    No averaging/trending features - only raw context data
    """
    
    def __init__(self, max_recent_matches: int = 3):
        """
        Initialize enhanced feature engineer
        
        Args:
            max_recent_matches: Number of recent matches for raw performance data (3 for stability)
        """
        self.max_recent_matches = max_recent_matches
        self.feature_columns = []
        self.feature_usage_log = {}
        
        logger.info(f"Initialized FeatureEngineer with 30-feature system (last {max_recent_matches} matches tracking)")
    
    def create_training_dataset(self, datasets: Dict[str, pd.DataFrame], 
                              training_cutoff_date: datetime, sample_size: int = 10000) -> pd.DataFrame:
        """
        Create training dataset with 30-feature system
        """
        logger.info(f"Creating training dataset with 30 features, cutoff date: {training_cutoff_date}")
        
        try:
            player_innings_df = datasets['player_innings']
            
            if 'fantasy_points' not in player_innings_df.columns:
                logger.error("fantasy_points column not found in player_innings data")
                raise ValueError("fantasy_points column is required for training")
            
            # Get historical matches BEFORE training cutoff date
            training_matches = player_innings_df[
                (player_innings_df['match_date'] < training_cutoff_date) &
                (player_innings_df['fantasy_points'].notna()) 
            ].sort_values('match_date')
            
            logger.info(f"Found {len(training_matches)} valid matches before {training_cutoff_date}")
            
            if len(training_matches) > sample_size:
                training_matches = training_matches.sample(sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} matches from available data")
            
            training_features = []
            processed_count = 0
            
            for idx, match_row in training_matches.iterrows():
                try:
                    # CRITICAL: Use data BEFORE this match date
                    feature_cutoff_date = match_row['match_date'] - timedelta(days=1)
                    
                    features = self._extract_features_for_match(
                        match_row, datasets, feature_cutoff_date=feature_cutoff_date
                    )
                    features['fantasy_points'] = match_row['fantasy_points']
                    training_features.append(features)
                    processed_count += 1
                    
                except Exception as e:
                    logger.debug(f"Skipping match: {e}")
                    continue
                
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} training samples...")
            
            if len(training_features) == 0:
                raise ValueError("No valid training features could be created")
            
            training_df = pd.DataFrame(training_features)
            logger.info(f"Created training dataset: {len(training_df)} samples, {len(training_df.columns)} features")
            
            # Log feature usage
            self._log_feature_usage(training_df)
            
            return training_df
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise
    
    def create_test_dataset(self, datasets: Dict[str, pd.DataFrame],
                           test_start_date: datetime, test_end_date: datetime) -> pd.DataFrame:
        """Create test dataset with proper temporal filtering"""
        logger.info(f"Creating test dataset: {test_start_date} to {test_end_date}")

        try:
            player_innings_df = datasets['player_innings']

            test_matches = player_innings_df[
                (player_innings_df['match_date'] >= test_start_date) &
                (player_innings_df['match_date'] < test_end_date) &
                (player_innings_df['fantasy_points'].notna())
            ].sort_values('match_date')

            logger.info(f"Found {len(test_matches)} test matches")

            test_features = []

            for idx, match_row in test_matches.iterrows():
                try:
                    feature_cutoff_date = match_row['match_date'] - timedelta(days=1)

                    features = self._extract_features_for_match(
                        match_row, datasets, feature_cutoff_date=feature_cutoff_date
                    )
                    features['fantasy_points'] = match_row['fantasy_points']
                    test_features.append(features)

                except Exception as e:
                    logger.debug(f"Skipping test match: {e}")
                    continue
        
            test_df = pd.DataFrame(test_features)
            logger.info(f"Created test dataset: {len(test_df)} samples")
        
            return test_df
        
        except Exception as e:
            logger.error(f"Error creating test dataset: {e}")
            raise

    def create_prediction_features(self, player_ids: List[int], venue_id: int,
                                 datasets: Dict[str, pd.DataFrame], 
                                 prediction_cutoff_date: datetime) -> pd.DataFrame:
        """Create 30-feature predictions for given players"""
        logger.info(f"Creating 30-feature predictions for {len(player_ids)} players with cutoff: {prediction_cutoff_date}")
        
        try:
            prediction_features = []
            
            for player_id in player_ids:
                try:
                    mock_match = {
                        'player_id': player_id,
                        'ground_id': venue_id,
                        'match_date': prediction_cutoff_date + timedelta(days=1),
                        'team_id': None,
                        'opposition_id': None
                    }
                    
                    features = self._extract_features_for_match(
                        mock_match, datasets, feature_cutoff_date=prediction_cutoff_date
                    )
                    prediction_features.append(features)
                    
                except Exception as e:
                    logger.warning(f"Could not create features for player {player_id}: {e}")
                    default_features = self._get_default_features(player_id)
                    prediction_features.append(default_features)
            
            prediction_df = pd.DataFrame(prediction_features)
            logger.info(f"Created prediction features: {len(prediction_df)} players using data up to {prediction_cutoff_date}")
            
            return prediction_df
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            raise
    
    def _extract_features_for_match(self, match_row: Dict, datasets: Dict[str, pd.DataFrame],
                                  feature_cutoff_date: datetime) -> Dict:
        """Extract 30-feature system with proper date filtering"""
        player_id = match_row['player_id']
        venue_id = match_row.get('ground_id', 0)
        opposition_id = match_row.get('opposition_id', 0)
        
        features = {'player_id': player_id}
        
        # Get player's historical data with strict date filtering
        all_player_data = datasets['player_innings'][
            datasets['player_innings']['player_id'] == player_id
        ]
        
        player_data = all_player_data[
            all_player_data['match_date'] < feature_cutoff_date
        ].sort_values('match_date')
        
        logger.debug(f"Player {player_id}: Using {len(player_data)} matches before {feature_cutoff_date}")
        
        # Apply temporal modeling strategy (early season uses previous year, mid/late uses current year)
        current_year = feature_cutoff_date.year
        current_year_data = player_data[player_data['match_date'].dt.year == current_year]
        
        if len(current_year_data) <= 2:
            # Use all available data for early season
            working_data = player_data
            archetype_data = player_data[player_data['match_date'].dt.year == current_year - 1]  # Previous year for archetypes
            logger.debug(f"Player {player_id}: Early season, using all available data ({len(working_data)} matches)")
        else:
            # Use current year only for mid/late season
            working_data = current_year_data
            archetype_data = current_year_data
            logger.debug(f"Player {player_id}: Mid/late season, using current year only ({len(working_data)} matches)")
        
        # Extract 30 features
        
        # Features 1-10: Recent Performance Context
        features.update(self._extract_recent_performance_features(working_data))
        
        # Features 11-15: Player Archetype (Current year priority)
        features.update(self._extract_player_archetype_features(archetype_data, current_year))
        
        # Features 16-20: Ball-by-Ball Context
        features.update(self._extract_ball_by_ball_features(player_id, working_data, datasets.get('deliveries')))
        
        # Features 21-24: Opposition Context
        features.update(self._extract_opposition_context_features(opposition_id, feature_cutoff_date, datasets))
        
        # Features 25-28: Venue Context
        features.update(self._extract_venue_context_features(venue_id, datasets.get('venue_stats')))
        
        # Features 29-30: Match Context
        features.update(self._extract_match_context_features(player_id, venue_id, datasets))
        
        return features
    
    def _extract_recent_performance_features(self, player_data: pd.DataFrame) -> Dict:
        """
        Extract recent performance features (Features 1-10)
        Raw values only - no averages or trends
        """
        features = {}
        
        # Default values for missing data
        default_runs = 20  # Conservative default
        default_wickets = 0
        default_out = 1  # Assume out by default
        
        # Features 1-3: Runs last 3 matches (raw values)
        for i in range(1, 4):
            if len(player_data) >= i:
                features[f'runs_match_{i}_ago'] = float(player_data.iloc[-i]['runs_batting'])
                self.feature_usage_log[f'runs_match_{i}_ago'] = 'used_real_data'
            else:
                features[f'runs_match_{i}_ago'] = default_runs
                self.feature_usage_log[f'runs_match_{i}_ago'] = 'used_default'
        
        # Features 4-6: Wickets last 3 matches (raw values)
        for i in range(1, 4):
            if len(player_data) >= i:
                features[f'wickets_match_{i}_ago'] = float(player_data.iloc[-i]['wickets_taken'])
                self.feature_usage_log[f'wickets_match_{i}_ago'] = 'used_real_data'
            else:
                features[f'wickets_match_{i}_ago'] = default_wickets
                self.feature_usage_log[f'wickets_match_{i}_ago'] = 'used_default'
        
        # Features 7-9: Was out last 3 matches (binary values)
        for i in range(1, 4):
            if len(player_data) >= i:
                # Convert 'out' column to binary (1 if out, 0 if not out)
                out_value = player_data.iloc[-i]['out']
                features[f'was_out_match_{i}_ago'] = 1 if pd.notna(out_value) and out_value != '' else 0
                self.feature_usage_log[f'was_out_match_{i}_ago'] = 'used_real_data'
            else:
                features[f'was_out_match_{i}_ago'] = default_out
                self.feature_usage_log[f'was_out_match_{i}_ago'] = 'used_default'
        
        # Feature 10: Data availability (count of available matches)
        features['data_availability'] = min(len(player_data), 3)
        self.feature_usage_log['data_availability'] = f'available_matches_{features["data_availability"]}'
        
        return features
    
    def _extract_player_archetype_features(self, player_data: pd.DataFrame, current_year: int) -> Dict:
        """
        Extract player archetype features (Features 11-15)
        Current year priority with fallback
        """
        features = {}
        
        if len(player_data) == 0:
            # Default archetype for players with no data
            features.update({
                'max_fantasy_score_current_year': 50.0,
                'min_fantasy_score_current_year': 5.0,
                'scores_over_60_count_current_year': 0,
                'scores_under_15_count_current_year': 0,
                'total_matches_current_year': 0
            })
            self.feature_usage_log.update({k: 'used_default' for k in features.keys()})
            return features
        
        fantasy_scores = player_data['fantasy_points'].dropna()
        
        if len(fantasy_scores) == 0:
            # No valid fantasy scores
            features.update({
                'max_fantasy_score_current_year': 50.0,
                'min_fantasy_score_current_year': 5.0,
                'scores_over_60_count_current_year': 0,
                'scores_under_15_count_current_year': 0,
                'total_matches_current_year': len(player_data)
            })
            self.feature_usage_log.update({k: 'used_default_no_fp' for k in features.keys()})
        else:
            # Features 11-15: Player archetype indicators
            features['max_fantasy_score_current_year'] = float(fantasy_scores.max())
            features['min_fantasy_score_current_year'] = float(fantasy_scores.min())
            features['scores_over_60_count_current_year'] = int((fantasy_scores > 60).sum())
            features['scores_under_15_count_current_year'] = int((fantasy_scores < 15).sum())
            features['total_matches_current_year'] = len(fantasy_scores)
            
            self.feature_usage_log.update({k: 'used_real_data' for k in features.keys()})
        
        return features
    
    def _extract_ball_by_ball_features(self, player_id: int, player_data: pd.DataFrame, 
                                      deliveries_df: pd.DataFrame = None) -> Dict:
        """
        Extract ball-by-ball context features (Features 16-20)
        """
        features = {}
        
        # Default values if no delivery data or recent match data
        default_features = {
            'powerplay_boundaries_last_match': 2,
            'death_over_runs_last_match': 15,
            'middle_over_dots_last_match': 8,
            'total_balls_faced_last_match': 20,
            'total_boundaries_last_match': 3
        }
        
        if deliveries_df is None or len(player_data) == 0:
            features.update(default_features)
            self.feature_usage_log.update({k: 'used_default_no_delivery_data' for k in default_features.keys()})
            return features
        
        # Get last match details
        last_match = player_data.iloc[-1] if len(player_data) > 0 else None
        
        if last_match is None:
            features.update(default_features)
            self.feature_usage_log.update({k: 'used_default_no_last_match' for k in default_features.keys()})
            return features
        
        last_match_date = last_match['match_date']
        
        # Get deliveries for last match
        last_match_deliveries = deliveries_df[
            (deliveries_df['batter_id'] == player_id) &
            (deliveries_df['match_date'] == last_match_date)
        ] if deliveries_df is not None else pd.DataFrame()
        
        if len(last_match_deliveries) == 0:
            # Use batting stats from player_innings as proxy
            features['powerplay_boundaries_last_match'] = min(int(last_match.get('fours', 0) + last_match.get('sixes', 0)), 6)
            features['death_over_runs_last_match'] = min(int(last_match.get('runs_batting', 0) * 0.3), 40)  # Estimate 30% runs in death
            features['middle_over_dots_last_match'] = max(0, int(last_match.get('balls_faced', 20) * 0.4))  # Estimate 40% dots in middle
            features['total_balls_faced_last_match'] = int(last_match.get('balls_faced', 20))
            features['total_boundaries_last_match'] = int(last_match.get('fours', 0) + last_match.get('sixes', 0))
            
            self.feature_usage_log.update({k: 'used_proxy_from_innings_data' for k in ['powerplay_boundaries_last_match', 'death_over_runs_last_match', 'middle_over_dots_last_match', 'total_balls_faced_last_match', 'total_boundaries_last_match']})
        else:
            # Calculate from actual delivery data
            
            # Feature 16: Powerplay boundaries (overs 1-6)
            powerplay_deliveries = last_match_deliveries[last_match_deliveries['over'] <= 6]
            powerplay_boundaries = len(powerplay_deliveries[powerplay_deliveries['batsman_runs'].isin([4, 6])])
            features['powerplay_boundaries_last_match'] = powerplay_boundaries
            
            # Feature 17: Death over runs (overs 16-20)
            death_deliveries = last_match_deliveries[last_match_deliveries['over'] >= 16]
            death_runs = death_deliveries['batsman_runs'].sum()
            features['death_over_runs_last_match'] = int(death_runs)
            
            # Feature 18: Middle over dots (overs 7-15)
            middle_deliveries = last_match_deliveries[
                (last_match_deliveries['over'] >= 7) & (last_match_deliveries['over'] <= 15)
            ]
            middle_dots = len(middle_deliveries[middle_deliveries['batsman_runs'] == 0])
            features['middle_over_dots_last_match'] = middle_dots
            
            # Feature 19: Total balls faced
            features['total_balls_faced_last_match'] = len(last_match_deliveries)
            
            # Feature 20: Total boundaries
            total_boundaries = len(last_match_deliveries[last_match_deliveries['batsman_runs'].isin([4, 6])])
            features['total_boundaries_last_match'] = total_boundaries
            
            self.feature_usage_log.update({k: 'used_real_delivery_data' for k in ['powerplay_boundaries_last_match', 'death_over_runs_last_match', 'middle_over_dots_last_match', 'total_balls_faced_last_match', 'total_boundaries_last_match']})
        
        return features
    
    def _extract_opposition_context_features(self, opposition_id: int, feature_cutoff_date: datetime, 
                                           datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Extract opposition context features (Features 21-24)
        """
        features = {}
        
        # Default opposition features
        default_features = {
            'opposition_runs_conceded_last_match': 160,
            'opposition_wickets_taken_last_match': 6,
            'opposition_total_bowlers_used_recent': 5,
            'opposition_matches_won_last_5': 2
        }
        
        if opposition_id == 0 or 'player_innings' not in datasets:
            features.update(default_features)
            self.feature_usage_log.update({k: 'used_default_no_opposition_data' for k in default_features.keys()})
            return features
        
        # Get opposition team's recent matches
        opposition_matches = datasets['player_innings'][
            (datasets['player_innings']['team_id'] == opposition_id) &
            (datasets['player_innings']['match_date'] < feature_cutoff_date)
        ].sort_values('match_date')
        
        if len(opposition_matches) == 0:
            features.update(default_features)
            self.feature_usage_log.update({k: 'used_default_no_opposition_matches' for k in default_features.keys()})
            return features
        
        # Get last match bowling performance
        last_match_date = opposition_matches['match_date'].iloc[-1]
        last_match_bowling = opposition_matches[
            (opposition_matches['match_date'] == last_match_date) &
            (opposition_matches['innings_bowling'].notna())
        ]
        
        # Feature 21: Opposition runs conceded last match
        if len(last_match_bowling) > 0:
            features['opposition_runs_conceded_last_match'] = int(last_match_bowling['runs_conceded'].sum())
            self.feature_usage_log['opposition_runs_conceded_last_match'] = 'used_real_data'
        else:
            features['opposition_runs_conceded_last_match'] = default_features['opposition_runs_conceded_last_match']
            self.feature_usage_log['opposition_runs_conceded_last_match'] = 'used_default_no_bowling_data'
        
        # Feature 22: Opposition wickets taken last match
        if len(last_match_bowling) > 0:
            features['opposition_wickets_taken_last_match'] = int(last_match_bowling['wickets_taken'].sum())
            self.feature_usage_log['opposition_wickets_taken_last_match'] = 'used_real_data'
        else:
            features['opposition_wickets_taken_last_match'] = default_features['opposition_wickets_taken_last_match']
            self.feature_usage_log['opposition_wickets_taken_last_match'] = 'used_default_no_bowling_data'
        
        # Feature 23: Total bowlers used recently
        recent_bowlers = opposition_matches.tail(20)  # Last ~4 matches
        unique_bowlers = len(recent_bowlers[recent_bowlers['innings_bowling'].notna()]['player_id'].unique())
        features['opposition_total_bowlers_used_recent'] = min(unique_bowlers, 8)  # Cap at 8
        self.feature_usage_log['opposition_total_bowlers_used_recent'] = 'used_real_data' if unique_bowlers > 0 else 'used_default'
        
        # Feature 24: Opposition matches won in last 5
        if 'matchstat' in datasets:
            recent_results = datasets['matchstat'][
                (datasets['matchstat']['team_id'] == opposition_id) &
                (datasets['matchstat']['match_date'] < feature_cutoff_date)
            ].tail(5)
            
            wins = len(recent_results[recent_results['result'] == 'won'])
            features['opposition_matches_won_last_5'] = wins
            self.feature_usage_log['opposition_matches_won_last_5'] = 'used_real_data'
        else:
            features['opposition_matches_won_last_5'] = default_features['opposition_matches_won_last_5']
            self.feature_usage_log['opposition_matches_won_last_5'] = 'used_default_no_matchstat'
        
        return features
    
    def _extract_venue_context_features(self, venue_id: int, venue_stats_df: pd.DataFrame = None) -> Dict:
        """
        Extract venue context features (Features 25-28)
        """
        features = {}
        
        # Default venue features
        default_features = {
            'venue_total_matches': 50,
            'venue_highest_score': 200,
            'venue_lowest_score': 120,
            'venue_total_boundaries': 800
        }
        
        if venue_stats_df is None or len(venue_stats_df) == 0:
            features.update(default_features)
            self.feature_usage_log.update({k: 'used_default_no_venue_data' for k in default_features.keys()})
            return features
        
        # Find venue data
        venue_data = venue_stats_df[venue_stats_df['venue_id'] == venue_id]
        
        if len(venue_data) == 0:
            # Use average from all venues
            features['venue_total_matches'] = int(venue_stats_df['total_matches_venue'].mean())
            features['venue_highest_score'] = int(venue_stats_df['highest_score_venue'].mean())
            features['venue_lowest_score'] = int(venue_stats_df['lowest_score_venue'].mean())
            features['venue_total_boundaries'] = int(venue_stats_df.get('total_boundaries_venue', default_features['venue_total_boundaries']).mean())
            
            self.feature_usage_log.update({k: 'used_venue_average' for k in ['venue_total_matches', 'venue_highest_score', 'venue_lowest_score', 'venue_total_boundaries']})
        else:
            venue = venue_data.iloc[0]
            
            # Features 25-28: Raw venue characteristics
            features['venue_total_matches'] = int(venue.get('total_matches_venue', default_features['venue_total_matches']))
            features['venue_highest_score'] = int(venue.get('highest_score_venue', default_features['venue_highest_score']))
            features['venue_lowest_score'] = int(venue.get('lowest_score_venue', default_features['venue_lowest_score']))
            features['venue_total_boundaries'] = int(venue.get('total_boundaries_venue', default_features['venue_total_boundaries']))
            
            self.feature_usage_log.update({k: 'used_real_venue_data' for k in ['venue_total_matches', 'venue_highest_score', 'venue_lowest_score', 'venue_total_boundaries']})
        
        return features
    
    def _extract_match_context_features(self, player_id: int, venue_id: int, 
                                      datasets: Dict[str, pd.DataFrame]) -> Dict:
        """
        Extract match context features (Features 29-30)
        """
        features = {}
        
        # Feature 29: High scoring venue (binary)
        venue_stats = datasets.get('venue_stats')
        if venue_stats is not None and len(venue_stats) > 0:
            venue_data = venue_stats[venue_stats['venue_id'] == venue_id]
            if len(venue_data) > 0:
                highest_score = venue_data.iloc[0].get('highest_score_venue', 180)
                features['high_scoring_venue'] = 1 if highest_score > 200 else 0
                self.feature_usage_log['high_scoring_venue'] = 'used_real_venue_data'
            else:
                features['high_scoring_venue'] = 0  # Conservative default
                self.feature_usage_log['high_scoring_venue'] = 'used_default_no_venue'
        else:
            features['high_scoring_venue'] = 0
            self.feature_usage_log['high_scoring_venue'] = 'used_default_no_venue_data'
        
        # Feature 30: Player career matches at venue
        player_innings = datasets.get('player_innings')
        if player_innings is not None:
            player_venue_matches = player_innings[
                (player_innings['player_id'] == player_id) &
                (player_innings['ground_id'] == venue_id)
            ]
            features['player_career_matches_at_venue'] = len(player_venue_matches)
            self.feature_usage_log['player_career_matches_at_venue'] = 'used_real_data'
        else:
            features['player_career_matches_at_venue'] = 0
            self.feature_usage_log['player_career_matches_at_venue'] = 'used_default_no_innings_data'
        
        return features
    
    def _get_default_features(self, player_id: int) -> Dict:
        """Default features for players with no historical data"""
        default_features = {
            'player_id': player_id,
            # Recent performance features (1-10)
            'runs_match_1_ago': 20.0, 'runs_match_2_ago': 20.0, 'runs_match_3_ago': 20.0,
            'wickets_match_1_ago': 0.0, 'wickets_match_2_ago': 0.0, 'wickets_match_3_ago': 0.0,
            'was_out_match_1_ago': 1, 'was_out_match_2_ago': 1, 'was_out_match_3_ago': 1,
            'data_availability': 0,
            # Player archetype features (11-15)
            'max_fantasy_score_current_year': 50.0,
            'min_fantasy_score_current_year': 5.0,
            'scores_over_60_count_current_year': 0,
            'scores_under_15_count_current_year': 0,
            'total_matches_current_year': 0,
            # Ball-by-ball features (16-20)
            'powerplay_boundaries_last_match': 2,
            'death_over_runs_last_match': 15,
            'middle_over_dots_last_match': 8,
            'total_balls_faced_last_match': 20,
            'total_boundaries_last_match': 3,
            # Opposition context features (21-24)
            'opposition_runs_conceded_last_match': 160,
            'opposition_wickets_taken_last_match': 6,
            'opposition_total_bowlers_used_recent': 5,
            'opposition_matches_won_last_5': 2,
            # Venue context features (25-28)
            'venue_total_matches': 50,
            'venue_highest_score': 200,
            'venue_lowest_score': 120,
            'venue_total_boundaries': 800,
            # Match context features (29-30)
            'high_scoring_venue': 0,
            'player_career_matches_at_venue': 0
        }
        
        # Log all defaults
        self.feature_usage_log.update({k: 'used_default_debut_player' for k in default_features.keys() if k != 'player_id'})
        
        return default_features
    
    def _log_feature_usage(self, training_df: pd.DataFrame) -> None:
        """Log feature usage statistics"""
        logger.info("=== FEATURE USAGE SUMMARY ===")
        
        # Count feature usage types
        usage_counts = {}
        for feature, usage_type in self.feature_usage_log.items():
            if usage_type not in usage_counts:
                usage_counts[usage_type] = []
            usage_counts[usage_type].append(feature)
        
        for usage_type, features in usage_counts.items():
            logger.info(f"{usage_type}: {len(features)} features")
            logger.debug(f"  Features: {features}")
        
        # Check for missing features
        expected_features = [
            'runs_match_1_ago', 'runs_match_2_ago', 'runs_match_3_ago',
            'wickets_match_1_ago', 'wickets_match_2_ago', 'wickets_match_3_ago',
            'was_out_match_1_ago', 'was_out_match_2_ago', 'was_out_match_3_ago',
            'data_availability', 'max_fantasy_score_current_year', 'min_fantasy_score_current_year',
            'scores_over_60_count_current_year', 'scores_under_15_count_current_year',
            'total_matches_current_year', 'powerplay_boundaries_last_match',
            'death_over_runs_last_match', 'middle_over_dots_last_match',
            'total_balls_faced_last_match', 'total_boundaries_last_match',
            'opposition_runs_conceded_last_match', 'opposition_wickets_taken_last_match',
            'opposition_total_bowlers_used_recent', 'opposition_matches_won_last_5',
            'venue_total_matches', 'venue_highest_score', 'venue_lowest_score',
            'venue_total_boundaries', 'high_scoring_venue', 'player_career_matches_at_venue'
        ]
        
        missing_features = [f for f in expected_features if f not in training_df.columns]
        if missing_features:
            logger.warning(f"MISSING FEATURES: {missing_features}")
        else:
            logger.info("✅ All 30 expected features present in dataset")
        
        # Log data availability statistics
        if 'data_availability' in training_df.columns:
            data_avail_counts = training_df['data_availability'].value_counts().sort_index()
            logger.info("Data Availability Distribution:")
            for avail_level, count in data_avail_counts.items():
                logger.info(f"  {avail_level} recent matches: {count} players ({count/len(training_df)*100:.1f}%)")
    
    def add_team_context(self, features_df: pd.DataFrame, squad_team1: List[int], 
                        squad_team2: List[int]) -> pd.DataFrame:
        """Add team assignment context"""
        logger.info("Adding team context to features...")
        
        try:
            df = features_df.copy()
            
            df['is_team1'] = df['player_id'].isin(squad_team1)
            df['is_team2'] = df['player_id'].isin(squad_team2)
            df['team_id'] = 1
            df.loc[df['is_team2'], 'team_id'] = 2
            
            df['opponent_squad'] = df['player_id'].apply(
                lambda x: squad_team2 if x in squad_team1 else squad_team1 if x in squad_team2 else []
            )
            
            logger.info("Team context added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding team context: {e}")
            raise
    
    def prepare_features_for_training(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare 30 features for training"""
        logger.info("Preparing 30 features for ML training...")
        
        try:
            df = training_df.copy()
            
            # Remove rows with missing target
            initial_count = len(df)
            df = df.dropna(subset=['fantasy_points'])
            final_count = len(df)
            
            if initial_count > final_count:
                logger.info(f"Removed {initial_count - final_count} rows with missing fantasy_points")
            
            # Store feature column names (exclude target and metadata)
            exclude_cols = ['player_id', 'fantasy_points', 'opponent_squad', 'is_team1', 'is_team2', 'team_id']
            self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            logger.info(f"Training features prepared: {len(df)} samples, {len(self.get_feature_columns())} features")
            logger.info(f"30-Feature system ready: {self.get_feature_columns()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing training features: {e}")
            raise
    
    def prepare_features_for_prediction(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare 30 features for prediction"""
        logger.info("Preparing 30 features for prediction...")
    
        try:
            df = prediction_df.copy()
        
            # Remove metadata columns
            exclude_cols = ['opponent_squad', 'is_team1', 'is_team2', 'team_id']
            keep_cols = [col for col in df.columns if col not in exclude_cols]
            df = df[keep_cols]
        
            logger.info(f"Prediction features prepared: {len(df)} players, {len(df.columns)} features")
        
            return df
        
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            raise
    
    def get_feature_columns(self) -> List[str]:
        """Get list of 30 feature columns (excluding target and metadata)"""
        exclude_cols = ['player_id', 'fantasy_points', 'opponent_squad', 'is_team1', 'is_team2', 'team_id']
        return [col for col in self.feature_columns if col not in exclude_cols]
    
    def get_feature_usage_summary(self) -> Dict[str, str]:
        """Get feature usage log for debugging"""
        return self.feature_usage_log.copy()