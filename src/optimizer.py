"""
POINT-BASED Fantasy Cricket Team Optimizer
Creates exactly ONE valid team with NO duplicate players and proper constraints
Optimized for point-based prediction system with 46-feature Dream11 components
Includes starting probability calculation based on recent team matches
NO CREDIT CONSTRAINTS - Focus on best team regardless of cost
"""

import pandas as pd
import numpy as np
import logging
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus, value
from typing import Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class FantasyTeamOptimizer:
    """
    POINT-BASED optimizer that creates exactly one valid fantasy cricket team
    - NO duplicate players
    - Exactly 11 players
    - One captain, one vice-captain
    - Proper role constraints
    - Starting probability consideration
    - Optimized for point-based predictions
    - NO CREDIT CONSTRAINTS
    """
    
    def __init__(self, max_per_team: int = 7, min_starting_probability: float = 0.5):
        self.max_per_team = max_per_team
        self.min_starting_probability = min_starting_probability
        
        # Standard fantasy cricket constraints
        self.min_wicketkeepers = 1
        self.max_wicketkeepers = 4
        self.min_batters = 3
        self.max_batters = 6
        self.min_allrounders = 1
        self.max_allrounders = 4
        self.min_bowlers = 3
        self.max_bowlers = 6
        
        logger.info(f"Initialized Point-Based FantasyTeamOptimizer (NO CREDITS)")
        logger.info(f"Max per team: {max_per_team}")
        logger.info(f"Min starting probability: {min_starting_probability}")
        logger.info("Optimized for 46-feature point-based Dream11 component predictions")
    
    def calculate_starting_probability_year_specific(self, player_id: int, match_date: datetime, 
                                                   player_innings_df: pd.DataFrame) -> float:
        """
        Calculate year-specific starting probability using improved logic
        Compatible with point-based feature engineering temporal strategy
        """
        try:
            year = match_date.year

            # Step 1: Get player's recent team_id before match_date (same year only)
            # This aligns with point-based feature engineer's temporal modeling
            df_before = player_innings_df[
                (player_innings_df['player_id'] == player_id) & 
                (player_innings_df['match_date'] < match_date) & 
                (player_innings_df['match_date'].dt.year == year)
            ]
            df_before = df_before.sort_values('match_date', ascending=False)

            if df_before.empty:
                # Check previous year data (similar to feature engineer's archetype fallback)
                df_prev_year = player_innings_df[
                    (player_innings_df['player_id'] == player_id) & 
                    (player_innings_df['match_date'].dt.year == year - 1)
                ]
                
                if df_prev_year.empty:
                    logger.debug(f"No historical data for player {player_id}, assuming 0.6 probability")
                    return 0.6  # Slightly optimistic for point-based system
                else:
                    logger.debug(f"Using previous year data for player {player_id} starting probability")
                    recent_team_id = df_prev_year.sort_values('match_date', ascending=False).iloc[0]['team_id']
            else:
                recent_team_id = df_before.iloc[0]['team_id']

            # Step 2: Get all matches of that team before match_date (current year focus)
            team_matches = player_innings_df[
                (player_innings_df['team_id'] == recent_team_id) & 
                (player_innings_df['match_date'] < match_date) & 
                (player_innings_df['match_date'].dt.year == year)
            ]

            # Step 3: Get unique matches by match_id with date
            team_matches_unique = team_matches[['match_id','match_date']].drop_duplicates()
            team_matches_unique = team_matches_unique.sort_values('match_date', ascending=False)

            # Step 4: Get last 3 matches of the team (consistent with point-based max_recent_matches=3)
            last3_team_matches = team_matches_unique.head(3)

            if len(last3_team_matches) == 0:
                logger.debug(f"No recent team matches found for player {player_id}, assuming 0.6 probability")
                return 0.6

            # Step 5: Check if player was in squad for each match
            results = []
            for _, row in last3_team_matches.iterrows():
                mid = row['match_id']
                # Check if player present in this match
                played = not player_innings_df[
                    (player_innings_df['match_id'] == mid) & 
                    (player_innings_df['player_id'] == player_id)
                ].empty
                results.append(int(played))

            # Step 6: Calculate probability with point-based adjustment
            base_probability = np.mean(results) if results else 0.6
            
            # Adjust probability based on data availability (similar to point-based features)
            data_availability = len(results)
            if data_availability >= 3:
                # High confidence in probability calculation
                probability = base_probability
            elif data_availability >= 2:
                # Medium confidence - slight adjustment toward league average
                probability = 0.7 * base_probability + 0.3 * 0.6
            else:
                # Low confidence - more conservative
                probability = 0.5 * base_probability + 0.5 * 0.6

            logger.debug(f"Player {player_id}: played {sum(results)}/{len(results)} recent matches in {year}, probability: {probability:.2f}")
            return probability

        except Exception as e:
            logger.warning(f"Error calculating starting probability for player {player_id}: {e}")
            return 0.6  # Default for point-based system
    
    def add_starting_probabilities(self, candidate_df: pd.DataFrame, 
                                 match_date: datetime, 
                                 player_innings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add starting probabilities to candidate DataFrame
        Enhanced for point-based prediction system
        
        Args:
            candidate_df: DataFrame with point-based player predictions
            match_date: Date of upcoming match
            player_innings_df: DataFrame with historical player data
            
        Returns:
            Enhanced DataFrame with starting_probability column
        """
        logger.info("Calculating starting probabilities for point-based candidates...")
        
        enhanced_df = candidate_df.copy()
        starting_probs = []
        
        for _, player in candidate_df.iterrows():
            player_id = player['player_id']
            prob = self.calculate_starting_probability_year_specific(player_id, match_date, player_innings_df)
            starting_probs.append(prob)
        
        enhanced_df['starting_probability'] = starting_probs
        
        # Log probability distribution with point-based context
        prob_stats = enhanced_df['starting_probability'].describe()
        logger.info(f"Point-based starting probability stats: mean={prob_stats['mean']:.2f}, min={prob_stats['min']:.2f}, max={prob_stats['max']:.2f}")
        
        # Count players above minimum threshold
        likely_starters = (enhanced_df['starting_probability'] >= self.min_starting_probability).sum()
        logger.info(f"Point-based candidates with starting probability >= {self.min_starting_probability}: {likely_starters}/{len(enhanced_df)}")
        
        # Analyze by data quality if available (from point-based predictions)
        if 'data_quality' in enhanced_df.columns:
            quality_prob = enhanced_df.groupby('data_quality')['starting_probability'].agg(['mean', 'count'])
            logger.info("Starting probability by data quality:")
            for quality, stats in quality_prob.iterrows():
                logger.info(f"  {quality}: {stats['count']} players, avg prob = {stats['mean']:.2f}")
        
        return enhanced_df
    
    def optimize_single_team(self, candidate_df: pd.DataFrame, 
                           match_date: datetime = None, 
                           player_innings_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, int, int]:
        """
        Create exactly ONE valid fantasy team with NO duplicates
        Optimized for point-based prediction system with 46 Dream11 component features
        NO CREDIT CONSTRAINTS - Focus on best possible team
        
        Args:
            candidate_df: DataFrame with point-based player predictions
            match_date: Date of upcoming match (for starting probability)
            player_innings_df: Historical player data (for starting probability)
            
        Returns:
            team_df: DataFrame with exactly 11 unique players
            captain_id: Single captain
            vice_captain_id: Single vice-captain
        """
        logger.info(f"Optimizing single team from {len(candidate_df)} point-based candidates (NO CREDITS)...")
        
        # CRITICAL: Ensure no duplicate players in input
        if candidate_df['player_id'].duplicated().any():
            logger.warning("Removing duplicate players from point-based candidate list...")
            candidate_df = candidate_df.drop_duplicates(subset=['player_id']).reset_index(drop=True)
            logger.info(f"After deduplication: {len(candidate_df)} unique players")
        
        # Ensure we have the required prediction column (from point-based predictor)
        pred_col = None
        for col in ['fp_pred', 'predicted_fantasy_points']:
            if col in candidate_df.columns:
                pred_col = col
                break
        
        if pred_col is None:
            raise ValueError("No prediction column found. Expected 'fp_pred' or 'predicted_fantasy_points'")
        
        # Standardize prediction column name
        if pred_col != 'fp_pred':
            candidate_df['fp_pred'] = candidate_df[pred_col]
        
        # Add starting probabilities if data is provided
        if match_date is not None and player_innings_df is not None:
            candidate_df = self.add_starting_probabilities(candidate_df, match_date, player_innings_df)
            
            # Filter out players with very low starting probability
            before_filter = len(candidate_df)
            candidate_df = candidate_df[candidate_df['starting_probability'] >= self.min_starting_probability]
            after_filter = len(candidate_df)
            
            if before_filter != after_filter:
                logger.info(f"Filtered out {before_filter - after_filter} point-based candidates with low starting probability")
        else:
            logger.info("No starting probability calculation (missing match_date or player_innings_df)")
            candidate_df['starting_probability'] = 1.0  # Assume all likely to start
        
        # Validate input columns
        required_cols = ['player_id', 'fp_pred', 'team_id', 'role']
        for col in required_cols:
            if col not in candidate_df.columns:
                raise ValueError(f"Missing required column for point-based optimization: {col}")
        
        # Standardize roles
        candidate_df['role_std'] = candidate_df['role'].apply(self._standardize_role)
        
        # Enhanced candidate analysis for point-based system
        logger.info("Point-based candidate analysis (NO CREDITS):")
        logger.info(f"- Prediction range: {candidate_df['fp_pred'].min():.1f} to {candidate_df['fp_pred'].max():.1f}")
        logger.info(f"- Starting prob range: {candidate_df['starting_probability'].min():.2f} to {candidate_df['starting_probability'].max():.2f}")
        logger.info(f"- Role distribution: {candidate_df['role_std'].value_counts().to_dict()}")
        logger.info(f"- Team distribution: {candidate_df['team_id'].value_counts().to_dict()}")
        
        # Point-based specific analysis
        if 'point_category' in candidate_df.columns:
            logger.info(f"- Point categories: {candidate_df['point_category'].value_counts().to_dict()}")
        if 'data_quality' in candidate_df.columns:
            logger.info(f"- Data quality: {candidate_df['data_quality'].value_counts().to_dict()}")
        if 'prediction_confidence' in candidate_df.columns:
            conf_stats = candidate_df['prediction_confidence'].describe()
            logger.info(f"- Confidence range: {conf_stats['min']:.2f} to {conf_stats['max']:.2f} (mean: {conf_stats['mean']:.2f})")
        
        # Check if optimization is possible
        self._validate_feasibility(candidate_df)
        
        try:
            # Create optimization problem
            prob = LpProblem("Point_Based_Fantasy_Team_No_Credits", LpMaximize)
            
            # Get unique players
            players = candidate_df['player_id'].tolist()
            player_data = candidate_df.set_index('player_id').to_dict('index')
            
            # Decision variables - binary selection for each player
            x = LpVariable.dicts('select', players, cat=LpBinary)
            
            # POINT-BASED OBJECTIVE: maximize predicted fantasy points weighted by starting probability
            # Enhanced weighting for point-based predictions with confidence
            objective_terms = []
            for p in players:
                base_points = player_data[p]['fp_pred']
                start_prob = player_data[p]['starting_probability']
                
                # Add confidence bonus if available (from point-based predictor)
                confidence_bonus = 1.0
                if 'prediction_confidence' in player_data[p]:
                    confidence_bonus = 1.0 + (player_data[p]['prediction_confidence'] - 0.5) * 0.2
                
                # Point-based weighted score
                weighted_score = base_points * start_prob * confidence_bonus
                objective_terms.append(weighted_score * x[p])
            
            prob += lpSum(objective_terms)
            
            # CONSTRAINT 1: Exactly 11 players (NO MORE, NO LESS)
            prob += lpSum([x[p] for p in players]) == 11
            
            # REMOVED CONSTRAINT 2: Credits limit - NO LONGER APPLIED
            
            # CONSTRAINT 3: Max players per team
            team_1_players = [p for p in players if player_data[p]['team_id'] == 1]
            team_2_players = [p for p in players if player_data[p]['team_id'] == 2]
            
            if team_1_players:
                prob += lpSum([x[p] for p in team_1_players]) <= self.max_per_team
            if team_2_players:
                prob += lpSum([x[p] for p in team_2_players]) <= self.max_per_team
            
            # CONSTRAINT 4: Role constraints
            for role in ['Wicketkeeper', 'Batter', 'Allrounder', 'Bowler']:
                role_players = [p for p in players if player_data[p]['role_std'] == role]
                if role_players:
                    min_count, max_count = self._get_role_limits(role)
                    prob += lpSum([x[p] for p in role_players]) >= min_count
                    prob += lpSum([x[p] for p in role_players]) <= max_count
            
            # CONSTRAINT 5: Enhanced starting probability constraint for point-based system
            # Ensure team has good mix of reliable starters
            high_prob_players = [p for p in players if player_data[p]['starting_probability'] >= 0.8]
            if len(high_prob_players) >= 6:  # If enough high-probability players available
                prob += lpSum([x[p] for p in high_prob_players]) >= 6
            
            # Solve
            logger.info("Solving point-based optimization (NO CREDITS)...")
            prob.solve()
            
            if LpStatus[prob.status] != 'Optimal':
                raise RuntimeError(f"Point-based optimization failed: {LpStatus[prob.status]}")
            
            # Extract selected players
            selected_player_ids = [p for p in players if value(x[p]) > 0.5]
            
            # CRITICAL VALIDATION
            if len(selected_player_ids) != 11:
                raise RuntimeError(f"Wrong team size: {len(selected_player_ids)} (expected 11)")
            
            if len(set(selected_player_ids)) != len(selected_player_ids):
                raise RuntimeError("Duplicate players detected in point-based optimization!")
            
            # Create team DataFrame
            team_df = candidate_df[candidate_df['player_id'].isin(selected_player_ids)].copy()
            
            # Enhanced captain/vice-captain selection for point-based system
            # Priority: High starting probability AND high predicted points AND high confidence
            team_df['captain_score'] = (
                team_df['fp_pred'] * 
                team_df['starting_probability'] * 
                team_df.get('prediction_confidence', 1.0)
            )
            
            team_df = team_df.sort_values('captain_score', ascending=False).reset_index(drop=True)
            
            # Select captain and vice-captain from top performers with good starting probability
            likely_starters = team_df[team_df['starting_probability'] >= 0.6]
            if len(likely_starters) >= 2:
                captain_id = int(likely_starters.iloc[0]['player_id'])
                vice_captain_id = int(likely_starters.iloc[1]['player_id'])
            else:
                # Fall back to top predictions regardless of starting probability
                captain_id = int(team_df.iloc[0]['player_id'])
                vice_captain_id = int(team_df.iloc[1]['player_id'])
            
            # Final validation
            self._validate_final_team(team_df, captain_id, vice_captain_id)
            
            # Point-based logging with enhanced metrics (NO CREDITS)
            total_points = team_df['fp_pred'].sum()
            weighted_points = (team_df['fp_pred'] * team_df['starting_probability']).sum()
            avg_starting_prob = team_df['starting_probability'].mean()
            avg_confidence = team_df.get('prediction_confidence', pd.Series([1.0]*len(team_df))).mean()
            
            logger.info("POINT-BASED TEAM OPTIMIZATION SUCCESSFUL (NO CREDITS)!")
            logger.info(f"- Players: {len(team_df)} unique")
            logger.info(f"- Total predicted points: {total_points:.1f}")
            logger.info(f"- Weighted predicted points: {weighted_points:.1f}")
            logger.info(f"- Average starting probability: {avg_starting_prob:.2f}")
            logger.info(f"- Average prediction confidence: {avg_confidence:.2f}")
            
            captain_prob = team_df[team_df['player_id']==captain_id]['starting_probability'].iloc[0]
            vc_prob = team_df[team_df['player_id']==vice_captain_id]['starting_probability'].iloc[0]
            logger.info(f"- Captain: {captain_id} (start prob: {captain_prob:.2f})")
            logger.info(f"- Vice-Captain: {vice_captain_id} (start prob: {vc_prob:.2f})")
            
            # Point-based specific insights
            if 'point_category' in team_df.columns:
                category_dist = team_df['point_category'].value_counts().to_dict()
                logger.info(f"- Point categories in team: {category_dist}")
            
            if 'data_quality' in team_df.columns:
                quality_dist = team_df['data_quality'].value_counts().to_dict()
                logger.info(f"- Data quality in team: {quality_dist}")
            
            return team_df, captain_id, vice_captain_id
            
        except Exception as e:
            logger.error(f"Point-based optimization error: {e}")
            raise
    
    def _standardize_role(self, role: str) -> str:
        """Standardize role names for point-based system"""
        role_upper = str(role).upper()
        
        if any(x in role_upper for x in ['WK', 'KEEPER', 'WICKET']):
            return 'Wicketkeeper'
        elif any(x in role_upper for x in ['BAT', 'BATS']):
            return 'Batter'
        elif any(x in role_upper for x in ['ALL', 'AR', 'ROUND']):
            return 'Allrounder'
        elif any(x in role_upper for x in ['BOWL', 'BWL']):
            return 'Bowler'
        else:
            return 'Batter'  # Default
    
    def _get_role_limits(self, role: str) -> Tuple[int, int]:
        """Get min/max limits for each role"""
        limits = {
            'Wicketkeeper': (self.min_wicketkeepers, self.max_wicketkeepers),
            'Batter': (self.min_batters, self.max_batters),
            'Allrounder': (self.min_allrounders, self.max_allrounders),
            'Bowler': (self.min_bowlers, self.max_bowlers)
        }
        return limits.get(role, (0, 11))
    
    def _validate_feasibility(self, candidate_df: pd.DataFrame):
        """Check if point-based optimization is mathematically possible"""
        role_counts = candidate_df['role_std'].value_counts()
        
        # Check each role has minimum required players
        issues = []
        for role in ['Wicketkeeper', 'Batter', 'Allrounder', 'Bowler']:
            min_required, _ = self._get_role_limits(role)
            available = role_counts.get(role, 0)
            if available < min_required:
                issues.append(f"{role}: need {min_required}, have {available}")
        
        if issues:
            raise ValueError(f"Point-based optimization infeasible constraints: {', '.join(issues)}")
        
        # Enhanced feasibility checks for point-based system
        total_players = len(candidate_df)
        if total_players < 11:
            raise ValueError(f"Not enough players for optimization: {total_players} (need 11)")
        
        # Check team distribution
        team_counts = candidate_df['team_id'].value_counts()
        for team_id, count in team_counts.items():
            max_from_team = min(count, self.max_per_team)
            if sum(team_counts) - count + max_from_team < 11:
                raise ValueError(f"Not enough players available with team limits")
        
        # Check starting probability constraints
        likely_starters = (candidate_df['starting_probability'] >= self.min_starting_probability).sum()
        if likely_starters < 11:
            logger.warning(f"Only {likely_starters} players meet minimum starting probability threshold")
            logger.warning("Consider lowering min_starting_probability or adding more candidates")
        
        logger.info("✓ Point-based feasibility validation passed (NO CREDITS)")
    
    def _validate_final_team(self, team_df: pd.DataFrame, captain_id: int, vice_captain_id: int):
        """Validate final point-based team meets all constraints"""
        # Check team size
        if len(team_df) != 11:
            raise RuntimeError(f"Invalid point-based team size: {len(team_df)}")
        
        # Check no duplicates
        if team_df['player_id'].duplicated().any():
            raise RuntimeError("Final point-based team has duplicate players!")
        
        # Check captain/vice-captain are in team
        player_ids = set(team_df['player_id'])
        if captain_id not in player_ids:
            raise RuntimeError("Captain not in point-based team!")
        if vice_captain_id not in player_ids:
            raise RuntimeError("Vice-captain not in point-based team!")
        if captain_id == vice_captain_id:
            raise RuntimeError("Captain and vice-captain are same player!")
        
        # Check role constraints
        role_counts = team_df['role_std'].value_counts()
        for role in ['Wicketkeeper', 'Batter', 'Allrounder', 'Bowler']:
            count = role_counts.get(role, 0)
            min_req, max_req = self._get_role_limits(role)
            if count < min_req or count > max_req:
                raise RuntimeError(f"Role constraint violated in point-based team: {role} has {count} (need {min_req}-{max_req})")
        
        # Check team distribution
        team_counts = team_df['team_id'].value_counts()
        for team_id, count in team_counts.items():
            if count > self.max_per_team:
                raise RuntimeError(f"Too many players from team {team_id}: {count} (max {self.max_per_team})")
        
        logger.info("✓ Final point-based team validation passed (NO CREDITS)")
    
    def print_team_details(self, team_df: pd.DataFrame, captain_id: int, vice_captain_id: int):
        """Print detailed team information with point-based analysis (NO CREDITS)"""
        print("\n" + "="*85)
        print("POINT-BASED FANTASY CRICKET TEAM (46-Feature Dream11 Components) - NO CREDITS")
        print("="*85)
        
        print("\nSELECTED PLAYING XI:")
        print(f"{'#':<3} | {'Player ID':<10} | {'Name':<15} | {'Role':<12} | {'Team':<5} | {'Points':<7} | {'Start%':<6} | {'Conf':<5} | {'Category':<12}")
        print("-" * 85)
        
        for i, (_, player) in enumerate(team_df.iterrows(), 1):
            player_id = player['player_id']
            name = str(player.get('name', f'Player_{player_id}'))[:15]
            role = player.get('role_std', player.get('role', 'Unknown'))[:12]
            team = f"T{player['team_id']}"
            points = player['fp_pred']
            start_prob = player.get('starting_probability', 1.0) * 100
            confidence = player.get('prediction_confidence', 1.0)
            category = str(player.get('point_category', 'Unknown'))[:12]
            
            # Mark captain and vice-captain
            mark = ""
            if player_id == captain_id:
                mark = " (C)"
                name = name[:12] + mark
            elif player_id == vice_captain_id:
                mark = " (VC)"
                name = name[:12] + mark
            
            print(f"{i:<3} | {player_id:<10} | {name:<15} | {role:<12} | {team:<5} | {points:<7.1f} | {start_prob:<6.0f} | {confidence:<5.2f} | {category:<12}")
        
        # Point-based enhanced summary statistics (NO CREDITS)
        total_points = team_df['fp_pred'].sum()
        weighted_points = (team_df['fp_pred'] * team_df['starting_probability']).sum()
        avg_starting_prob = team_df['starting_probability'].mean()
        avg_confidence = team_df.get('prediction_confidence', pd.Series([1.0]*len(team_df))).mean()
        role_dist = team_df['role_std'].value_counts().to_dict()
        team_dist = team_df['team_id'].value_counts().to_dict()
        
        print(f"\nPOINT-BASED TEAM SUMMARY (NO CREDITS):")
        print(f"- Total Predicted Points: {total_points:.1f}")
        print(f"- Weighted Predicted Points: {weighted_points:.1f}")
        print(f"- Average Starting Probability: {avg_starting_prob:.1%}")
        print(f"- Average Prediction Confidence: {avg_confidence:.1%}")
        print(f"- Role Distribution: {role_dist}")
        print(f"- Team Distribution: {team_dist}")
        print(f"- Captain: Player {captain_id}")
        print(f"- Vice-Captain: Player {vice_captain_id}")
        
        # Point-based specific insights
        high_prob_players = (team_df['starting_probability'] >= 0.8).sum()
        high_conf_players = (team_df.get('prediction_confidence', pd.Series([1.0]*len(team_df))) >= 0.8).sum()
        print(f"- High Confidence Starters (≥80%): {high_prob_players}/11")
        print(f"- High Confidence Predictions (≥80%): {high_conf_players}/11")
        
        # Performance category distribution
        if 'point_category' in team_df.columns:
            category_dist = team_df['point_category'].value_counts().to_dict()
            print(f"- Performance Categories: {category_dist}")
        
        # Data quality distribution
        if 'data_quality' in team_df.columns:
            quality_dist = team_df['data_quality'].value_counts().to_dict()
            print(f"- Data Quality: {quality_dist}")
        
        print("="*85)


def create_point_based_fantasy_team(candidate_df: pd.DataFrame, 
                                  match_date: datetime = None,
                                  player_innings_df: pd.DataFrame = None,
                                  max_per_team: int = 7,
                                  min_starting_probability: float = 0.5) -> Tuple[pd.DataFrame, int, int]:
    """
    Simple function to create a single valid fantasy team with point-based predictions
    Optimized for 46-feature Dream11 component system
    NO CREDIT CONSTRAINTS - Focus on best possible team
    
    Args:
        candidate_df: DataFrame with point-based predictions (must include fp_pred, team_id, role, player_id)
        match_date: Date of upcoming match for starting probability calculation
        player_innings_df: Historical data for starting probability calculation
        max_per_team: Maximum players from one team (default: 7)
        min_starting_probability: Minimum starting probability threshold (default: 0.5)
    
    Returns:
        team_df: DataFrame with exactly 11 unique players
        captain_id: Player ID of captain
        vice_captain_id: Player ID of vice-captain
    """
    optimizer = FantasyTeamOptimizer(
        max_per_team=max_per_team,
        min_starting_probability=min_starting_probability
    )
    return optimizer.optimize_single_team(candidate_df, match_date, player_innings_df)


# Legacy function name for backward compatibility
def create_enhanced_fantasy_team(candidate_df: pd.DataFrame, 
                               match_date: datetime = None,
                               player_innings_df: pd.DataFrame = None,
                               credits_limit: float = 100.0,  # Ignored parameter for compatibility
                               max_per_team: int = 7,
                               min_starting_probability: float = 0.5) -> Tuple[pd.DataFrame, int, int]:
    """
    Legacy function name - redirects to point-based team creation
    NOTE: credits_limit parameter is ignored (NO CREDIT CONSTRAINTS)
    """
    logger.info("Using legacy function name - redirecting to point-based team creation (NO CREDITS)")
    return create_point_based_fantasy_team(
        candidate_df=candidate_df,
        match_date=match_date, 
        player_innings_df=player_innings_df,
        max_per_team=max_per_team,
        min_starting_probability=min_starting_probability
    )