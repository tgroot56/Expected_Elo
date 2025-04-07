import os
import pandas as pd

class ELODataset:
    def __init__(self, season_ids, elo_dir="data/elo_rankings", conversion_map=None):
        """
        Initialize the ELODataset by loading the corresponding ELO CSV files for each season.
        
        Parameters:
            season_ids (list of str): Season identifiers, e.g., ['2022-2023', '2023-2024'].
            elo_dir (str): Directory where the ELO ranking files are stored.
                Files are expected to be named as: "<second_year>-06-01.csv" 
                (e.g., for '2023-2024', file is "2024-06-01.csv").
            conversion_map (dict): A mapping to convert team names from team_stats to the names used in the ELO CSV files.
        """
        self.season_ids = season_ids
        self.elo_dir = elo_dir
        self.elo_data = {}
        # For each season, automatically construct the file path and load the CSV.
        for season in season_ids:
            try:
                # For season "YYYY-YYYY", we use the second year.
                second_year = season.split('-')[1]
            except IndexError:
                raise ValueError("Season id must be in format 'YYYY-YYYY'.")
            file_name = f"{second_year}-06-01.csv"
            file_path = os.path.join(elo_dir, file_name)
            # Load the CSV file for that season.
            self.elo_data[season] = pd.read_csv(file_path)
        
        # Set default conversion map if not provided.
        if conversion_map is None:
            self.conversion_map = {
                "Luton Town": "Luton",
                "Manchester City": "Man City",
                "Manchester Utd": "Man United",
                "Newcastle Utd": "Newcastle",
                "Nott'ham Forest": "Forest",
                "Sheffield Utd": "Sheffield United",
                "Leeds United": "Leeds",
                "Leicester City": "Leicester",
                "Clermont Foot": "Clermont",
                "Paris S": "Paris SG"
            }
        else:
            self.conversion_map = conversion_map

    def get_ratings(self, team_stats_df):
        """
        Given a team_stats DataFrame that includes columns 'team_name' and 'season_id',
        retrieve the corresponding Elo rating from the loaded ELO data.
        
        Returns:
            A DataFrame with columns: team_name, season_id, Elo.
        """
        results = []
        for _, row in team_stats_df.iterrows():
            club = row["meta_data_team_name"]
            season = row["season_id"]
            # Get the ELO DataFrame for this season.
            elo_df = self.elo_data.get(season)
            if elo_df is not None:
                # Convert the club name if needed.
                elo_club = self.conversion_map.get(club, club)
                rating_series = elo_df.loc[elo_df["Club"] == elo_club, "Elo"]
                rating = rating_series.iloc[0] if not rating_series.empty else None
            else:
                rating = None
            results.append({
                "meta_data_team_name": club,
                "season_id": season,
                "Elo": rating
            })
        return pd.DataFrame(results)
    
    def merge_ratings(self, team_stats_df):
        """
        Merge Elo ratings with the given team_stats DataFrame.
        
        Returns:
            A new DataFrame with Elo ratings merged on 'team_name' and 'season_id'.
        """
        ratings_df = self.get_ratings(team_stats_df)
        merged_df = team_stats_df.merge(ratings_df, on=["meta_data_team_name", "season_id"], how="left")
        return merged_df
