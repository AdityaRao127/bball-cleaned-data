import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

def scrape_team_rankings(metric):
    base_url = "https://www.teamrankings.com/nba/"
    
    # Map of metrics
    url_mapping = {
        'abl': 'stat/average-biggest-lead',
        'ast_pp': 'stat/assists-per-game',
        'average_scoring_margin': 'stat/average-scoring-margin',
        'blk_pct': 'stat/blocks-per-game',
        'defensive_efficiency': 'stat/defensive-efficiency',
        'drb_pct': 'stat/defensive-rebounding-percentage',
        'efg_pct': 'stat/effective-field-goal-pct',
        'flr_pct': 'stat/floor-percentage',
        'ftr': 'stat/free-throw-rate',
        'opp_flr_pct': 'stat/opponent-floor-percentage',
        'opponent_efg_pct': 'stat/opponent-effective-field-goal-pct',
        'orb_pct': 'stat/offensive-rebounding-pct',
        'pfs_pct': 'stat/personal-fouls-per-game',
        'sht_pct': 'stat/shooting-pct',
        'stls_pdp': 'stat/steals-per-defensive-play',
        'tov_pct': 'stat/turnover-pct',
        'win_pct': 'stat/win-pct-all-games'
    }
    
    url = base_url + url_mapping[metric]
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the stats table
        table = soup.find('table', {'class': 'datatable'})
        if not table:
            return None
            
        data = []
        for row in table.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 7: 
                team = clean_team_name(cols[1].text.strip())
                rank = int(cols[0].text.strip())
                
                current = float(cols[2].text.strip().replace('%', ''))
                last_3 = float(cols[3].text.strip().replace('%', ''))
                home = float(cols[5].text.strip().replace('%', ''))
                away = float(cols[6].text.strip().replace('%', ''))
                
                weighted_stat = (
                    0.60 * current +
                    0.15 * last_3 +
                    0.125 * home +
                    0.125 * away
                )
                
                data.append({
                    'Team': team,
                    'Rank': rank,
                    'Statistic': round(weighted_stat, 3),
                    'Year': datetime.now().year
                })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error scraping {metric}: {str(e)}")
        return None

def clean_team_name(team):
    team_mapping = {
        'Atlanta Hawks': 'Atlanta',
        'Boston Celtics': 'Boston',
        'Brooklyn Nets': 'Brooklyn',
        'Charlotte Hornets': 'Charlotte',
        'Chicago Bulls': 'Chicago',
        'Cleveland Cavaliers': 'Cleveland',
        'Dallas Mavericks': 'Dallas',
        'Denver Nuggets': 'Denver',
        'Detroit Pistons': 'Detroit',
        'Golden State Warriors': 'Golden State',
        'Houston Rockets': 'Houston',
        'Indiana Pacers': 'Indiana',
        'Los Angeles Clippers': 'LA Clippers',
        'Los Angeles Lakers': 'LA Lakers',
        'Memphis Grizzlies': 'Memphis',
        'Miami Heat': 'Miami',
        'Milwaukee Bucks': 'Milwaukee',
        'Minnesota Timberwolves': 'Minnesota',
        'New Orleans Pelicans': 'New Orleans',
        'New York Knicks': 'New York',
        'Oklahoma City Thunder': 'Okla City',
        'Orlando Magic': 'Orlando',
        'Philadelphia 76ers': 'Philadelphia',
        'Phoenix Suns': 'Phoenix',
        'Portland Trail Blazers': 'Portland',
        'Sacramento Kings': 'Sacramento',
        'San Antonio Spurs': 'San Antonio',
        'Toronto Raptors': 'Toronto',
        'Utah Jazz': 'Utah',
        'Washington Wizards': 'Washington'
    }
    return team_mapping.get(team, team)

def create_current_stats():
    metrics = [
        'abl', 'ast_pp', 'average_scoring_margin', 'blk_pct',
        'defensive_efficiency', 'drb_pct', 'efg_pct', 'flr_pct',
        'ftr', 'opp_flr_pct', 'opponent_efg_pct', 'orb_pct',
        'pfs_pct', 'sht_pct', 'stls_pdp', 'tov_pct', 'win_pct'
    ]
    
    base_dir = "./Current_Data"
    os.makedirs(base_dir, exist_ok=True)
    
    print("Fetching Win Percentage data...")
    win_pct_df = scrape_team_rankings('win_pct')
    if win_pct_df is not None:
        win_pct_dict = win_pct_df.set_index('Team')['Statistic'].to_dict()
    else:
        print("Failed to fetch Win Percentage data.")
        win_pct_dict = {}
    
    for metric in metrics:
        print(f"Processing {metric}...")
        metric_dir = os.path.join(base_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        
        df = scrape_team_rankings(metric)
        
        if df is not None:
            df['Win Percentage'] = df['Team'].map(win_pct_dict)
            
            for team, team_data in df.groupby('Team'):
                team_file = os.path.join(metric_dir, f'{team}.csv')
                team_data[['Rank', 'Statistic', 'Year', 'Win Percentage']].to_csv(team_file, index=False)
                print(f"Created {team_file}")
        else:
            print(f"Failed to scrape data for {metric}")

if __name__ == "__main__":
    create_current_stats()
