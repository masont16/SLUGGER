# SLUGGER
 Statistical Lineup and Game Generated Evaluation for Runs

This program is made to predict the number of runs a team will score in today's MLB game. The model uses statistics from 2020 and onward and automatically updates after each day. The chosen algorithm for the model is a Random Forest, as I saw the highest mean squared error and r2 scores from this algorithm. 

When the program is run, the user should input the 3 letter abbreviation of the team they wish to gather a prediction for. 

The model can only predict the current day's schedule and will  return an error when the starting pitcher has not been announced yet. There will be future updates to solve these issues.

The spreadsheet of predictions and results will be updated daily, keeping a running count of profit if a user were to wager 1 unit on each bet since 7/24/2024.
