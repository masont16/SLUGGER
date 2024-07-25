import mlbstatsapi
import numpy as np
import statsapi
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import pandas as pd
from pybaseball import team_game_logs

# This function reads in the season game logs
# and pairs them with the stats of the starting pitcher
# for the given game
def getPitcherForSeason(frame, season, pitcher):
    df = pd.DataFrame()
    firstInit = []
    for i in frame.index:
        temp = frame['OppStart'][i]
        id1 = temp.index('(')
        oppStarter = temp[2:id1]
        firstInit.append(temp[0])
        # gets the stats of the starting pitcher
        tempdf = pitcher.loc[(pitcher['Name'].str.contains(oppStarter)) & (pitcher['team'] == frame['Opp'][i]) & (pitcher['Year'] == season) & (pitcher['GS'] != '0') & (pitcher['Name'].str[0] == temp[0])]
        tempdf.reset_index(drop=True, inplace=True)
        # if there are two starting pitchers with the same name on the same
        # team, the pitcher with the least starts in a season will be dropped
        if len(tempdf.index) == 2:
            maxIndex = max(tempdf['GS'][0], tempdf['GS'][1])
            tempdf = tempdf.loc[tempdf['GS'] == maxIndex]
        df = df._append(tempdf)
    df['firstInit'] = firstInit
    return df
mlb = mlbstatsapi.Mlb()
brefteamAbbreviations = ['LAA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'MIA', 'HOU', 'KCR', 'LAD', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
teamAbbreviations = ['LAA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL', 'DET', 'MIA', 'HOU', 'KCR', 'LAD', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH']
team = input('Give team abbreviation')
# gets all games on the current day's schedule
games = mlb.get_schedule(datetime.today().strftime('%Y-%m-%d'))
dates = games.dates
allPitch = pd.DataFrame()
# Used to get the pitching stats for all teams in the league in the past 5 years (same data is stored in csv for ease of use)
'''for team in brefteamAbbreviations:
    stts = team_pitching_bref(team, 2020, 2024)
    allPitch = allPitch._append(stts, ignore_index=True)
    for i in stts.index:
        teamName.append(team)
allPitch['team'] = teamName
allPitch.to_csv('pitcherStats.csv')'''
# Reads in the pitcher stats from csv created above
allPitch = pd.read_csv('pitcherStats.csv')
# Saves all game logs for the given team for the past 5 seasons
homeGameLog = team_game_logs(2023, team)
homeGameLog2 = team_game_logs(2024, team)
homeGameLog3 = team_game_logs(2022, team)
homeGameLog4 = team_game_logs(2021, team)
homeGameLog5 = team_game_logs(2020, team)
totalHomeLog = pd.concat([homeGameLog5, homeGameLog4, homeGameLog3, homeGameLog, homeGameLog2], ignore_index=True)
# Gets the starting pitcher's season stats in order of season schedule
pitch2020 = getPitcherForSeason(homeGameLog5, 2020, allPitch)
pitch2021 = getPitcherForSeason(homeGameLog4, 2021, allPitch)
pitch2022 = getPitcherForSeason(homeGameLog3, 2022, allPitch)
pitch2023 = getPitcherForSeason(homeGameLog, 2023, allPitch)
pitch2024 = getPitcherForSeason(homeGameLog2, 2024, allPitch)
awayPitch = pd.concat([pitch2020, pitch2021, pitch2022, pitch2023, pitch2024], ignore_index=True)
# Drops unnecessary columns and pairs the game logs to the pitcher's season stats
awayPitch = awayPitch.drop(columns=['Pos', 'Name', 'Year', 'Age', 'W', 'L', 'W-L%', 'G', 'GS', 'GF', 'CG', 'SHO', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'IBB', 'SO', 'HBP', 'BK', 'WP', 'BF', 'ERA', 'team'])
combined = pd.concat([totalHomeLog, awayPitch], axis=1)
combined.to_csv('allStatsandPitch.csv')
combined = combined.drop(columns=['Game', 'Rslt', 'Date', 'Opp', 'AB', 'IBB', 'ROE', 'NumPlayers', 'OppStart', 'PA', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'HBP', 'SH', 'SF', 'GDP', 'SB', 'CS', 'LOB', 'firstInit', 'Unnamed: 0', 'SO/W'])
combined = combined.dropna(subset=['ERA+'])
# Encodes throwing arm (left or right) as a binary value
encodedThr = pd.get_dummies(combined['Thr'])
combined['Thr'] = encodedThr['R']
combined = combined.rename(columns={'Thr': 'RHP'})
# Sets x and y values for regression
x = combined.values
y = combined['R'].values
# Deletes runs column from x
x = np.delete(x, 1, axis=1)
clf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0)
clf.fit(x, y)
# Automated pulling season stats for batting team and pitching stats for away team
stats = ['season']
groups = ['hitting']
# Finds game selected and compares 3 letter abbreviaton from BBref and MLB.com
for date in dates:
    for game in date.games:
        # sets home or away, batting stats, and pitcher id
        if brefteamAbbreviations.index(team) == teamAbbreviations.index(statsapi.get('team', {'teamId': game.teams.home.team.id})['teams'][0]['abbreviation']):
            home = True
            batting = mlb.get_team_stats(game.teams.home.team.id, stats=stats, groups=groups)
            gameid = statsapi.next_game(game.teams.home.team.id)
            pitcher = mlb.get_person(mlb.get_game(gameid).__dict__['gamedata'].__dict__['probablepitchers'].__dict__['away'].__dict__['id'])
        if brefteamAbbreviations.index(team) == teamAbbreviations.index(statsapi.get('team', {'teamId': game.teams.away.team.id})['teams'][0]['abbreviation']):
            home = False
            batting = mlb.get_team_stats(game.teams.away.team.id, stats=stats, groups=groups)
            gameid = statsapi.next_game(game.teams.away.team.id)
            pitcher = mlb.get_person(mlb.get_game(gameid).__dict__['gamedata'].__dict__['probablepitchers'].__dict__['home'].__dict__['id'])
# Sets the handedness of the opposing pitcher
pitchhand = pitcher.__dict__['pitchhand'].__dict__['code']
pitcher = pitcher.__dict__['namefirstlast']
tempdf = allPitch.loc[(allPitch['Name'].str.contains(pitcher)) & (allPitch['Year'] == 2024)]
tempdf2 = allPitch.loc[(allPitch['Name'].str.contains(pitcher)) & (allPitch['Year'] == 2023)]
# Gathers pitcher stats from last two seasons and averages them out.
# If pitcher is a rookie or didn't play last season, only 2024 stats will be used
if tempdf2.empty:
    FIP = tempdf['FIP']
    eraplus = tempdf['ERA+']
    WHIP = tempdf['WHIP']
    H9 = tempdf['H9']
    HR9 = tempdf['HR9']
    BB9 = tempdf['BB9']
    SO9 = tempdf['SO9']
elif tempdf.empty:
    FIP = tempdf2['FIP']
    eraplus = tempdf2['ERA+']
    WHIP = tempdf2['WHIP']
    H9 = tempdf2['H9']
    HR9 = tempdf2['HR9']
    BB9 = tempdf2['BB9']
    SO9 = tempdf2['SO9']
else:
    bothseasons = pd.concat([tempdf, tempdf2], ignore_index=True)
    FIP = (bothseasons['FIP'][0] + bothseasons['FIP'][1])/2
    eraplus = (bothseasons['ERA+'][0] + bothseasons['ERA+'][1])/2
    WHIP = (bothseasons['WHIP'][0] + bothseasons['WHIP'][1])/2
    H9 = (bothseasons['H9'][0] + bothseasons['H9'][1])/2
    HR9 = (bothseasons['HR9'][0] + bothseasons['HR9'][1])/2
    BB9 = (bothseasons['BB9'][0] + bothseasons['BB9'][1])/2
    SO9 = (bothseasons['SO9'][0] + bothseasons['SO9'][1])/2
hitting = batting['hitting']['season']
splits = hitting.splits[0].__dict__['stat'].__dict__
if pitchhand == 'L':
    RH = False
if pitchhand == 'R':
    RH = True
# Puts the predict data into a dataframe and uses the Random Forest Regressor
# model to predict the number of runs a team will score in today's MLB game
A=[home, splits['avg'],  splits['obp'], splits['slg'], splits['ops'], RH, eraplus, FIP, WHIP, H9, HR9, BB9, SO9]
tester = pd.DataFrame([A])
pred = clf.predict(tester)
print(pred)
