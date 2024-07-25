import mlbstatsapi
import numpy as np
import statsapi
import requests
from bs4 import BeautifulSoup as bs
from matplotlib import pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.model_selection import train_test_split
from datetime import datetime
import pandas as pd
from pybaseball import schedule_and_record, team_ids, team_batting_bref, team_game_logs, season_game_logs, playerid_reverse_lookup, team_pitching_bref, get_splits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

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
teamAbbreviations = ['LAA', 'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 'MIA', 'HOU', 'KCR', 'LAD', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN']
input = input('Please give the abbreviation of the team you would like to predict (Yankees = NYY)')
homeTeam = input
# gets all games on the current day's schedule
games = mlb.get_schedule(datetime.today().strftime('%Y-%m-%d'))
dates = games.dates
allPitch = pd.DataFrame()
# sorts through the games to find the home team id and the away starting pitcher
for date in dates:
    for game in date.games:
        if homeTeam == statsapi.get('team', {'teamId':game.teams.home.team.id})['teams'][0]['abbreviation']:
            awayTeam = statsapi.get('team', {'teamId':game.teams.away.team.id})['teams'][0]['abbreviation']
            homeid = game.teams.home.team.id
            ap = statsapi.get('game', {'gamePk': game.gamepk})['gameData']['probablePitchers']['away']
# Used to get the pitching stats for all teams in the league in the past 5 years (same data is stored in csv for ease of use)
'''for team in teamAbbreviations:
    stts = team_pitching_bref(team, 2020, 2024)
    allPitch = allPitch._append(stts, ignore_index=True)
    for i in stts.index:
        teamName.append(team)
allPitch['team'] = teamName
allPitch.to_csv('pitcherStats.csv')'''
# Reads in the pitcher stats from csv created above
allPitch = pd.read_csv('pitcherStats.csv')
# Saves all game logs for the given team for the past 5 seasons
homeGameLog = team_game_logs(2023, homeTeam)
homeGameLog2 = team_game_logs(2024, homeTeam)
homeGameLog3 = team_game_logs(2022, homeTeam)
homeGameLog4 = team_game_logs(2021, homeTeam)
homeGameLog5 = team_game_logs(2020, homeTeam)
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
# Manually set stats for prediction, will update and make this automatic in the next update
A=[False, '.257', '.329', '.421', '0.749', False, '101', '4.09', '1.305', '8', '1.2', '3.7', '10.3']
tester = pd.DataFrame([A])
pred = clf.predict(tester)
print(pred)
