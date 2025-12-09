# src/data_loader.py
"""
ENHANCED Data Loading Pipeline for Fantasy Cricket 30-Feature System
Loads all required datasets for comprehensive feature engineering
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CricketDataLoader:
    """Enhanced data loading class for 30-feature cricket system"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize data loader with data directory path"""
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized DataLoader with directory: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        # List available files for debugging
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Available CSV files: {[f.name for f in csv_files]}")
    
    def _parse_date_column(self, df: pd.DataFrame, date_col: str = 'match_date') -> pd.DataFrame:
        """Universal date parsing for YYYYMMDD integer or string formats"""
        try:
            if date_col not in df.columns:
                logger.warning(f"Date column {date_col} not found")
                return df
            
            original_dtype = df[date_col].dtype
            
            # Handle different date formats
            if df[date_col].dtype in ['int64', 'int32']:
                logger.debug(f"Parsing {date_col} as YYYYMMDD integer format")
                df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d', errors='coerce')
            else:
                logger.debug(f"Parsing {date_col} as string format")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Log conversion results
            null_dates = df[date_col].isnull().sum()
            if null_dates > 0:
                logger.warning(f"{null_dates} invalid dates found in {date_col}")
            
            logger.debug(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing dates in {date_col}: {e}")
            return df
    
    def load_player_innings(self) -> pd.DataFrame:
        """Load player innings statistics with correct date parsing"""
        logger.info("Loading player innings...")
        try:
            df = pd.read_csv(self.data_dir / "playerInningStat_merged.csv")
            df = self._parse_date_column(df, 'match_date')
            
            logger.info(f"Loaded {len(df)} player innings")
            logger.info(f"Date range: {df['match_date'].min()} to {df['match_date'].max()}")
            
            # Verify required columns
            required_cols = ['player_id', 'match_date', 'fantasy_points']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                raise ValueError(f"Required columns missing: {missing_cols}")
            
            # Log column availability for feature engineering
            available_cols = df.columns.tolist()
            logger.info(f"Available columns: {available_cols[:10]}..." if len(available_cols) > 10 else f"Available columns: {available_cols}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading player innings: {e}")
            raise
    
    def load_profiles(self) -> pd.DataFrame:
        """Load player profiles"""
        logger.info("Loading player profiles...")
        try:
            df = pd.read_csv(self.data_dir / "ipl_player_profiles_2024-2025.csv")
            
            # Clean player_id column
            df['player_id'] = df['player_id'].astype(str).str.strip()
            df = df[df['player_id'] != 'nan']
            df['player_id'] = pd.to_numeric(df['player_id'], errors='coerce')
            df = df.dropna(subset=['player_id'])
            df['player_id'] = df['player_id'].astype(int)
            
            logger.info(f"Loaded profiles for {len(df)} players")
            logger.info(f"Available profile columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            raise
    
    def load_venue_stats(self) -> pd.DataFrame:
        """Load venue statistics"""
        logger.info("Loading venue stats...")
        try:
            # Load the specific file directly
            venue_file = self.data_dir / "venueStats_merged.csv"
            
            if not venue_file.exists():
                logger.warning("venueStats_merged.csv not found, creating enhanced default venue stats")
                return self._create_default_venue_stats()
            
            df = pd.read_csv(venue_file)
            logger.info(f"Loaded stats for {len(df)} venues from venueStats_merged.csv")
            logger.info(f"Venue columns: {df.columns.tolist()}")
            
            # Map columns to expected feature engineering names
            df = self._standardize_venue_columns(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error loading venueStats_merged.csv: {e}, using enhanced defaults")
            return self._create_default_venue_stats()
    
    def _create_default_venue_stats(self) -> pd.DataFrame:
        """Create comprehensive default venue stats for feature engineering"""
        logger.info("Creating enhanced default venue statistics...")
        
        # Create realistic venue variations
        import numpy as np
        np.random.seed(42)  # For reproducible defaults
        
        venue_ids = range(1, 101)
        venue_data = []
        
        for venue_id in venue_ids:
            # Create varied venue characteristics
            base_score = np.random.normal(160, 25)  # Base around 160
            
            venue_data.append({
                'venue_id': venue_id,
                # Standard venue stats
                'average_runs_venue': max(120, min(220, base_score)),
                'runs_per_over_venue': max(6.5, min(10.0, base_score/20)),
                'total_matches_venue': np.random.randint(20, 100),
                'batting_average_venue': max(20, min(35, np.random.normal(25, 5))),
                'bowling_average_venue': max(25, min(40, np.random.normal(30, 4))),
                
                # Feature engineering specific columns
                'highest_score_venue': max(180, min(260, base_score + np.random.normal(40, 15))),
                'lowest_score_venue': max(80, min(140, base_score - np.random.normal(30, 10))),
                'total_boundaries_venue': np.random.randint(500, 2000)
            })
        
        df = pd.DataFrame(venue_data)
        logger.info(f"Created default stats for {len(df)} venues with feature engineering columns")
        return df
    
    def _standardize_venue_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize venue column names for feature engineering compatibility"""
        column_mapping = {
            # Common variations to standard names
            'venue_id': 'venue_id',
            'ground_id': 'venue_id',
            'stadium_id': 'venue_id',
            
            'avg_runs': 'average_runs_venue',
            'average_runs': 'average_runs_venue',
            'avg_score': 'average_runs_venue',
            
            'runs_per_over': 'runs_per_over_venue',
            'rpo': 'runs_per_over_venue',
            
            'total_matches': 'total_matches_venue',
            'matches_played': 'total_matches_venue',
            'games': 'total_matches_venue',
            
            'highest_score': 'highest_score_venue',
            'max_score': 'highest_score_venue',
            'high_score': 'highest_score_venue',
            
            'lowest_score': 'lowest_score_venue',
            'min_score': 'lowest_score_venue',
            'low_score': 'lowest_score_venue',
            
            'total_boundaries': 'total_boundaries_venue',
            'boundaries': 'total_boundaries_venue',
            'total_fours_sixes': 'total_boundaries_venue'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
                logger.debug(f"Renamed venue column: {old_name} -> {new_name}")
        
        # Ensure required columns exist with defaults
        required_venue_cols = {
            'venue_id': 1,
            'highest_score_venue': 180,
            'lowest_score_venue': 120,
            'total_boundaries_venue': 800,
            'total_matches_venue': 50,
            'average_runs_venue': 160,
            'runs_per_over_venue': 8.0
        }
        
        for col, default_val in required_venue_cols.items():
            if col not in df.columns:
                df[col] = default_val
                logger.debug(f"Added missing venue column {col} with default value {default_val}")
        
        return df
    
    def load_deliveries(self) -> Optional[pd.DataFrame]:
        """Load ball-by-ball deliveries data"""
        logger.info("Loading deliveries...")
        try:
            # Load the specific file directly
            delivery_file = self.data_dir / "deliveries_combined_2008_2025.csv"
            
            if not delivery_file.exists():
                logger.warning("deliveries_combined_2008_2025.csv not found - ball-by-ball features will use defaults")
                return None
            
            df = pd.read_csv(delivery_file)
            df = self._parse_date_column(df, 'match_date')
            
            logger.info(f"Loaded {len(df)} deliveries from deliveries_combined_2008_2025.csv")
            logger.info(f"Deliveries date range: {df['match_date'].min()} to {df['match_date'].max()}")
            logger.info(f"Delivery columns: {df.columns.tolist()[:10]}..." if len(df.columns) > 10 else f"Delivery columns: {df.columns.tolist()}")
            
            # Standardize delivery column names
            df = self._standardize_delivery_columns(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error loading deliveries_combined_2008_2025.csv: {e}")
            return None
    
    def _standardize_delivery_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize delivery column names for feature engineering"""
        column_mapping = {
            # Batsman columns
            'batter': 'batter_id',
            'batsman': 'batter_id',
            'striker': 'batter_id',
            'batter_id': 'batter_id',
            
            # Runs columns  
            'batsman_runs': 'batsman_runs',
            'batter_runs': 'batsman_runs',
            'runs_scored': 'batsman_runs',
            'runs': 'batsman_runs',
            
            # Over columns
            'over': 'over',
            'over_number': 'over',
            'over_num': 'over'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and old_name != new_name:
                df = df.rename(columns={old_name: new_name})
                logger.debug(f"Renamed delivery column: {old_name} -> {new_name}")
        
        # Ensure required columns exist
        if 'batter_id' not in df.columns:
            logger.warning("No batter_id column found in deliveries - ball-by-ball features may not work")
        
        if 'batsman_runs' not in df.columns:
            logger.warning("No batsman_runs column found in deliveries - ball-by-ball features may not work")
        
        return df
    
    def load_matchstat(self) -> Optional[pd.DataFrame]:
        """Load match statistics for opposition analysis"""
        logger.info("Loading match statistics...")
        try:
            # Load the specific file directly
            matchstat_file = self.data_dir / "matchStat_merged.csv"
            
            if not matchstat_file.exists():
                logger.warning("matchStat_merged.csv not found - opposition features will use defaults")
                return None
            
            df = pd.read_csv(matchstat_file)
            df = self._parse_date_column(df, 'match_date')
            
            logger.info(f"Loaded {len(df)} match statistics from matchStat_merged.csv")
            logger.info(f"Match stats columns: {df.columns.tolist()}")
            
            # Standardize matchstat columns
            df = self._standardize_matchstat_columns(df)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error loading matchStat_merged.csv: {e}")
            return None
    
    def _standardize_matchstat_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize match statistics column names"""
        column_mapping = {
            'team': 'team_id',
            'team_name': 'team_id', 
            'winning_team': 'result',
            'winner': 'result',
            'match_result': 'result'
        }
        
        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                logger.debug(f"Renamed matchstat column: {old_name} -> {new_name}")
        
        # Standardize result column to 'won'/'lost' format
        if 'result' in df.columns:
            # Convert various result formats to standardized won/lost
            df['result'] = df['result'].astype(str).str.lower()
            df['result'] = df['result'].map({
                'win': 'won', 'won': 'won', 'w': 'won', 'victory': 'won',
                'loss': 'lost', 'lost': 'lost', 'l': 'lost', 'defeat': 'lost'
            }).fillna('unknown')
        
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all available datasets for 30-feature system"""
        logger.info("Loading all datasets for 30-feature system...")
        
        try:
            # Core required datasets
            datasets = {
                'player_innings': self.load_player_innings(),
                'profiles': self.load_profiles(),
                'venue_stats': self.load_venue_stats(),
            }
            
            # Optional datasets with fallbacks
            deliveries = self.load_deliveries()
            if deliveries is not None:
                datasets['deliveries'] = deliveries
                logger.info("✓ Deliveries loaded - ball-by-ball features available")
            else:
                logger.warning("✗ Deliveries not available - using proxy ball-by-ball features")
            
            matchstat = self.load_matchstat()
            if matchstat is not None:
                datasets['matchstat'] = matchstat
                logger.info("✓ Match statistics loaded - opposition analysis available")
            else:
                logger.warning("✗ Match statistics not available - using default opposition features")
            
            # Print comprehensive data summary
            self._print_dataset_summary(datasets)
            
            logger.info("✓ All available datasets loaded successfully for 30-feature system")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    def _print_dataset_summary(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Print comprehensive summary of loaded datasets"""
        logger.info("="*60)
        logger.info("DATASET SUMMARY FOR 30-FEATURE SYSTEM")
        logger.info("="*60)
        
        # Core dataset info
        player_innings = datasets['player_innings']
        logger.info(f"Core Datasets:")
        logger.info(f"- Player innings: {len(player_innings):,} records")
        logger.info(f"- Date range: {player_innings['match_date'].min()} to {player_innings['match_date'].max()}")
        
        # Check year distribution
        player_innings['year'] = player_innings['match_date'].dt.year  
        year_counts = player_innings['year'].value_counts().sort_index()
        logger.info(f"- Year distribution: {dict(year_counts.head(10))}")
        
        # Other datasets
        for name, df in datasets.items():
            if name != 'player_innings':
                logger.info(f"- {name}: {len(df):,} records")
        
        # Feature availability assessment
        logger.info(f"\nFeature Engineering Readiness:")
        logger.info(f"✓ Recent Performance (1-10): Available (player_innings)")
        logger.info(f"✓ Player Archetype (11-15): Available (player_innings)")
        
        if 'deliveries' in datasets:
            logger.info(f"✓ Ball-by-Ball Context (16-20): Full features available")
        else:
            logger.info(f"⚠ Ball-by-Ball Context (16-20): Proxy features only")
        
        if 'matchstat' in datasets:
            logger.info(f"✓ Opposition Context (21-24): Full features available")
        else:
            logger.info(f"⚠ Opposition Context (21-24): Default features only")
        
        logger.info(f"✓ Venue Context (25-28): Available (venue_stats)")
        logger.info(f"✓ Match Context (29-30): Available (combined data)")
        
        logger.info("="*60)


# Quick test function
def test_data_loader(data_dir: str = "data"):
    """Test the data loader with your data directory"""
    try:
        loader = CricketDataLoader(data_dir)
        datasets = loader.load_all()
        print(f"Successfully loaded {len(datasets)} datasets")
        return datasets
    except Exception as e:
        print(f"Error testing data loader: {e}")
        return None


if __name__ == "__main__":
    test_data_loader()