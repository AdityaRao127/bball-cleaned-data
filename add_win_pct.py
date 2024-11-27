import os
import pandas as pd
import shutil

base_dir = './Cleaned_Data'
win_pct_dir = os.path.join(base_dir, 'win_pct')
output_dir = './New_Cleaned_Data'
os.makedirs(output_dir, exist_ok=True)

win_pct_data = {}

for file in os.listdir(win_pct_dir):
    if file.endswith('.csv'):
        team_name = file.split('.csv')[0]
        win_pct_path = os.path.join(win_pct_dir, file)
        win_pct_data[team_name] = pd.read_csv(win_pct_path)

for metric_dir in os.listdir(base_dir):
    if metric_dir == 'win_pct':
        continue

    metric_path = os.path.join(base_dir, metric_dir)
    if not os.path.isdir(metric_path):
        continue

    output_metric_path = os.path.join(output_dir, metric_dir)
    os.makedirs(output_metric_path, exist_ok=True)

    for file in os.listdir(metric_path):
        if file.endswith('.csv'):
            team_name = file.split('.csv')[0]
            stat_path = os.path.join(metric_path, file)

            # Read the current stat file and corresponding win_pct data
            stat_df = pd.read_csv(stat_path)
            win_pct_df = win_pct_data.get(team_name)

            if win_pct_df is not None:
                # Merge data on the year column
                # Selectively include only the 'year' and 'statistic' columns from win_pct_df
                win_pct_df = win_pct_df.rename(columns={'Statistic': 'Win Percentage'})[['Year', 'Win Percentage']]

                # Merge
                merged_df = stat_df.merge(win_pct_df, on='Year', how='left')

                # Save the updated CSV
                output_file = os.path.join(output_metric_path, file)
                merged_df.to_csv(output_file, index=False)

shutil.rmtree(base_dir)
os.rename(output_dir, base_dir)
