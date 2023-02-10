#Sports Betting Readme
###Introduction
This is a Python code to scrape sports betting data from the website basketball-reference.com. It uses the BeautifulSoup library to extract information from the site, and the Playwright library to fetch the HTML. The code creates a folder structure to store the data it collects.

I used the following work as a base for this project: https://github.com/dataquestio/project-walkthroughs/tree/master/nba_games
To this base I added the following functionality:
    1. Add and predict future games
    2. Add past sportsbook odds
    3. Make predictions on the over/under bet and spread bet for future games

###Requirements
To run this code, you will need to install the following packages:

pandas
numpy
playwright
BeautifulSoup

###Folder Structure
The code creates the following folder structure to store the data:

data: The root directory where the other folders are stored.
standings: This folder stores the standings pages for each season.
scores: This folder stores the box scores for each game.
temp: This folder stores the temporary box scores while they are being processed.
Code Overview
The code defines a range of seasons to scrape, from 2016 to 2023. If the data directory does not exist, the code creates the required folder structure. The code then uses the Playwright library to fetch the HTML for each season's standings and box scores. The information is then stored in the relevant folder. The code also includes a method to parse the HTML for the box scores, which extracts information such as the teams playing, the scores, and the date of the game.

###Conclusion
This code provides a simple way to scrape sports betting data from the website basketball-reference.com. By using the BeautifulSoup and Playwright libraries, the code can easily extract the required information and store it in a structured way. With this code, you will have the data you need to analyze the performance of various teams and make informed decisions when placing bets on sports games.