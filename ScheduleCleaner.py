import os
import csv

input_file = './nba_schedule_2024-25.csv'
input_dir = os.path.dirname(input_file)
output_dir = './Schedule'
os.makedirs(output_dir, exist_ok=True)

# match schedule .csv team names to project norm
team_name_formatter = {
    'Atlanta Hawks': 'Atlanta',
    'Boston Celtics' : 'Boston', 
    'Brooklyn Nets' : 'Brooklyn',
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

# create subfolders for each team's schedule inside output_dir
for entry in team_name_formatter:
    team_name = team_name_formatter[entry]
    team_folder = os.path.join(output_dir, f'{team_name}')
    os.makedirs(team_folder, exist_ok=True)
    csv_file_name = os.path.join(team_folder, f'{team_name}.csv')
    with open(csv_file_name, 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Date', 'Start Time (ET)', 'Opponent', 'Location (Home/Away)', 'Arena', 'Notes'])

with open(input_file, 'r') as schedule:
    csv_reader = csv.reader(schedule)
    # skip the first line
    next(csv_reader)
    for row in csv_reader:
        # put the data values into easy-use variables
        date = row[0]
        time = row[1]
        away_team = team_name_formatter[row[2]]
        home_team = team_name_formatter[row[3]]
        arena = row[4]
        note = row[5]
        home_team_folder = os.path.join(output_dir, home_team)
        home_team_csv = os.path.join(home_team_folder, f'{home_team}.csv')
        with open(home_team_csv, 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([date, time, away_team, 'H', arena, note])
        away_team_folder = os.path.join(output_dir, away_team)
        away_team_csv = os.path.join(away_team_folder, f'{away_team}.csv')
        with open(away_team_csv, 'a') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([date, time, home_team, 'A', arena, note])
    

# how to parse the input .csv
'''
note Date, Time, Home Team, Away Team, Arena
set team = Home Team
append to team.csv file (create if DNE):
    Date,Time,Away Team,H,Arena
set team = Away Team
append to team.csv file (create if DNE):
    Date,Time,Home Team,A,Arena
'''