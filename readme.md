# NBA Sport Betting

## Work in Progress:
Modeling predicts the winner of any given NBA game at 63% accuracy. I am working on building models to predict the OverUnder and Spread for given odds.

I used the code from the following github as a base and have heavily modified and added to it:
https://github.com/dataquestio/project-walkthroughs/tree/master/nba_games

## Goals:

The main goal of this project is to predict the outcomes of NBA games and determine how to bet on the moneyline, spread, and over/under. To achieve this goal, we will develop a machine learning model that analyzes various factors such as team performance, player stats, home/away records, and other relevant data.

My model will be trained on historical NBA game data to identify patterns and relationships that can help predict the outcome of future games. Once the model is trained, we will test its accuracy on a validation set of data and make any necessary adjustments to improve its performance.

After validating our model, we will use it to predict the outcomes of upcoming NBA games and provide recommendations on how to bet on the moneyline, spread, and over/under. These recommendations will be based on our model's predicted probabilities of each possible outcome, along with a consideration of the odds offered by bookmakers.

Overall, our goal is to build a reliable and accurate machine learning model that can help bettors make informed decisions and improve their chances of winning bets on NBA games.

## Installation
In order to run the script, the following Python libraries need to be installed:

pandas
numpy
BeautifulSoup
playwright
datetime
os
pickle

In addition to the Python libraries, Playwright browsers need to be installed by running playwright install on the command line or !playwright install from Jupyter.

## Usage
The script creates three directories: data/standings, data/scores, and data/temp. The data/standings directory contains HTML files with team standings data, the data/scores directory contains box score data for individual games, and the data/temp directory is used for temporary files.

Run script.py to webscrape recent games and make predictions on future games

## Conclusion
This project is still a work in progress. There are still bugs that exist, and I am currently working on adding features. That said, the model predicts future NBA games with an accuracy of 63% which is much better than simply predicting that the home team will win.