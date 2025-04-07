import time
import requests
import pandas as pd

class FBRAPIDataset:
    def __init__(self, api_key=None):
        # If no API key is provided, generate one.
        if api_key is None:
            self.api_key = self.get_fbrapi_key()
        else:
            self.api_key = api_key

    def get_fbrapi_key(self):
        """Get the FBR API Key."""
        time.sleep(6)
        response = requests.post('https://fbrapi.com/generate_api_key')
        return response.json()['api_key']

    def get_country_ids(self, target_countries):
        """Get the list of country codes from the FBR API."""
        time.sleep(6)
        url = "https://fbrapi.com/countries"
        headers = {"X-API-Key": self.api_key}
        response = requests.get(url, headers=headers)
        countries = response.json()

        country_codes = []
        for country in countries.get('data', []):
            if country.get('country') in target_countries:
                country_codes.append(country.get('country_code'))
        return country_codes

    def get_league_ids(self, target_leagues, country_codes):
        """Get the list of league IDs from the FBR API."""
        time.sleep(6)
        url = "https://fbrapi.com/leagues"
        headers = {"X-API-Key": self.api_key}
        league_ids = []
        for country_code in country_codes:
            params = {"country_code": country_code}
            response = requests.get(url, headers=headers, params=params)
            leagues = response.json()
            if 'data' in leagues:
                for league_type in leagues['data']:
                    for league in league_type.get('leagues', []):
                        if league.get('competition_name') in target_leagues:
                            league_ids.append(league.get('league_id'))
            time.sleep(6)
        return league_ids

    def get_season_ids(self, league_ids):
        """Get the list of season IDs from the FBR API."""
        time.sleep(6)
        url = "https://fbrapi.com/league-seasons"
        headers = {"X-API-Key": self.api_key}
        season_ids = {}
        for league_id in league_ids:
            params = {"league_id": league_id}
            response = requests.get(url, headers=headers, params=params)
            seasons = response.json()
            if 'data' in seasons:
                for season in seasons['data']:
                    if 'season_id' in season:
                        season_ids.setdefault(league_id, []).append(season['season_id'])
            time.sleep(6)
        return season_ids

    def fetch_team_stats(self, league_id, season_id):
        """Fetch team stats for a specific league and season from the FBR API."""
        time.sleep(6)
        url = "https://fbrapi.com/team-season-stats"
        headers = {"X-API-Key": self.api_key}
        params = {"league_id": league_id, "season_id": season_id}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            stats_data = response.json().get("data", [])
            for stat in stats_data:
                stat["league_id"] = league_id
                stat["season_id"] = season_id
            return stats_data
        else:
            print(f"Failed to fetch for league {league_id}, season {season_id}")
            return []

    @staticmethod
    def flatten_dict(d, parent_key='', sep='_'):
        """
        Recursively flattens a nested dictionary.
        For example, a nested key "stats" with a sub-key "matches_played" 
        becomes "stats_matches_played".
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(FBRAPIDataset.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
