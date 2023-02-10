#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from datetime import date
from datetime import timedelta
from datetime import datetime
import pickle


# In[2]:


SEASONS = list(range(2016,2024))

if os.path.exists('data'):
    pass
else:
    os.mkdir('data')
    os.mkdir(os.path.join('data', 'standings'))
    os.mkdir(os.path.join('data', 'scores'))
    os.mkdir(os.path.join('data', 'temp'))

DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
SCORES_DIR = os.path.join(DATA_DIR, "scores")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
standings_files = os.listdir(STANDINGS_DIR)

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
import time
# Make sure to install playwright browsers by running playwright install on the command line or !playwright install from Jupyter

async def get_html(url, selector, sleep=5, retries=3):
    html = None
    for i in range(1, retries+1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.firefox.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")
    
    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]
    
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            continue
        
        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)
            
async def scrape_game(standings_file):
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html)
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    for url in box_scores:
        save_path = os.path.join(TEMP_DIR, url.split("/")[-1])
        check_path_1 = os.path.join(SCORES_DIR, url.split("/")[-1])
        check_path_2 = os.path.join(TEMP_DIR, url.split("/")[-1])
        if os.path.exists(check_path_1) or os.path.exists(check_path_2):
            continue

        html = await get_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+") as f:
            f.write(html)
            
def parse_html(box_score):
    with open(box_score) as f:
        html = f.read()

    soup = BeautifulSoup(html)
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup

def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all('a')]
    season = os.path.basename(hrefs[1]).split("_")[0]
    return season

def read_line_score(soup):
    line_score = pd.read_html(str(soup), attrs = {'id': 'line_score'})[0]
    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols
    
    line_score = line_score[["team", "total"]]
    
    return line_score

def read_stats(soup, team, stat):
    df = pd.read_html(str(soup), attrs = {'id': f'box-{team}-game-{stat}'}, index_col=0)[0]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def read_line_score_test(soup):
    teams = soup.select('.scorebox')[0]
    hrefs = [t["href"] for t in teams.find_all('a')]
    team = [t for t in hrefs if '/teams' in t]
    first_team = team[0].split('/')[2]
    second_team = team[1].split('/')[2]
    scores = soup.find_all('div',{'class':'scores'})
    first_score = int(scores[0].text)
    second_score =int(scores[1].text)
    return pd.DataFrame({'team':[first_team,second_team],'total':[first_score,second_score]})
    


# In[3]:


## Delete old standings file and scrape new with most recent scores
month = datetime.now().strftime("%B").lower()
if os.path.exists(f'data/standings/NBA_2023_games-{month}.html'):
    os.remove(f'data/standings/NBA_2023_games-{month}.html')  ##Update month as needed
    scrape_season(2023)
else:
    scrape_season(2023)


# In[6]:


## run scrape_game function which opens each box score and saves it as an html file - skips if file already exists
for season in SEASONS:
    files = [s for s in standings_files if str(season) in s]
    
    for f in files:
        filepath = os.path.join(STANDINGS_DIR, f)
        
        scrape_game(filepath)


# In[7]:


box_scores = os.listdir(TEMP_DIR)
box_scores = [os.path.join(TEMP_DIR, f) for f in box_scores if f.endswith(".html")]
games = []
base_cols = None
for box_score in box_scores:
    soup = parse_html(box_score)

    line_score = read_line_score_test(soup)
    teams = list(line_score["team"])

    summaries = []
    for team in teams:
        basic = read_stats(soup, team, "basic")
        advanced = read_stats(soup, team, "advanced")

        totals = pd.concat([basic.iloc[-1,:], advanced.iloc[-1,:]])
        totals.index = totals.index.str.lower()

        maxes = pd.concat([basic.iloc[:-1].max(), advanced.iloc[:-1].max()])
        maxes.index = maxes.index.str.lower() + "_max"

        summary = pd.concat([totals, maxes])
        
        if base_cols is None:
            base_cols = list(summary.index.drop_duplicates(keep="first"))
            base_cols = [b for b in base_cols if "bpm" not in b]
        
        summary = summary[base_cols]
        
        summaries.append(summary)
    summary = pd.concat(summaries, axis=1).T

    game = pd.concat([summary, line_score], axis=1)

    game["home"] = [0,1]

    game_opp = game.iloc[::-1].reset_index()
    game_opp.columns += "_opp"

    full_game = pd.concat([game, game_opp], axis=1)
    full_game["season"] = read_season_info(soup)
    
    full_game["date"] = os.path.basename(box_score)[:8]
    full_game["date"] = pd.to_datetime(full_game["date"], format="%Y%m%d")
    
    full_game["won"] = full_game["total"] > full_game["total_opp"]
    games.append(full_game)
    
    if len(games) % 100 == 0:
        print(f"{len(games)} / {len(box_scores)}")


# In[8]:


try:
    temp = pd.concat(games, ignore_index=True)
    temp.to_csv("temp.csv")
    temp = pd.read_csv('temp.csv',index_col=0)
except:
    temp = pd.read_csv('temp.csv',index_col=0)
old = pd.read_csv("nba_games_updated.csv",index_col=0)
save = pd.concat([old,temp])
save.team = save.team.str.replace('CHO','CHA').str.replace('PHO','PHX').str.replace('BRK','BKN').str.replace('NJN','BKN').str.replace('NOH','NOP')
save.to_csv("nba_games_updated.csv")


# In[9]:


# gather all files
allfiles = os.listdir(TEMP_DIR)
 
# iterate on all files to move them to destination folder
for f in allfiles:
    src_path = os.path.join(TEMP_DIR, f)
    dst_path = os.path.join(SCORES_DIR, f)
    os.rename(src_path, dst_path)


# In[10]:


df = pd.read_csv('next_game.csv',index_col=0)
df.DATE = pd.to_datetime(df.DATE)
df = df[df.DATE>=datetime.today() - timedelta(days = 1)]
df = df.drop_duplicates(subset = 'team', keep='first')
df.to_csv('next_game_30.csv')


# In[11]:


df = pd.read_csv('nba_games_updated.csv', index_col=0)
df = df.sort_values("date")
df = df.reset_index(drop=True)
del df["mp.1"]
del df["mp_opp.1"]
del df["index_opp"]


# In[12]:


def add_target(group):
    group["target"] = group["won"].shift(-1)
    return group

df = df.groupby("team", group_keys=False).apply(add_target)

df["target"][pd.isnull(df["target"])] = 2
df["target"] = df["target"].astype(int, errors="ignore")

nulls = pd.isnull(df).sum()
nulls = nulls[nulls > 0]

valid_columns = df.columns[~df.columns.isin(nulls.index)]
df = df[valid_columns]


# In[13]:


from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier

rr = RidgeClassifier(alpha=1)

rf = RandomForestClassifier(max_depth=3, max_samples = .4, n_estimators = 100)

split = TimeSeriesSplit(n_splits=3)

sfs = SequentialFeatureSelector(rf, 
                                n_features_to_select=30, 
                                direction="backward",
                                cv=split,
                                n_jobs=1
                               )


# In[14]:


removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
selected_columns = df.columns[~df.columns.isin(removed_columns)]


# In[15]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])


# In[16]:


def backtest(data, model, predictors, target, start=2, step=1):
    all_predictions = []
    
    seasons = sorted(data["season"].unique())
    
    for i in range(start, len(seasons), step):
        season = seasons[i]
        train = data[data["season"] < season]
        test = data[data["season"] == season]
        
        model.fit(train[predictors], train[target])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test[target], preds], axis=1)
        combined.columns = ["actual", "prediction"]
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)


# In[17]:


df_rolling = df[list(selected_columns) + ["won", "team", "season"]]

def find_team_averages(team):
    rolling = team.rolling(10).mean()
    return rolling

df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

rolling_cols = [f"{col}_10" for col in df_rolling.columns]
df_rolling.columns = rolling_cols
df = pd.concat([df, df_rolling], axis=1)

df = df.dropna()


# In[18]:


def shift_col(team, col_name):
    next_col = team[col_name].shift(-1)
    return next_col

def add_col(df, col_name):
    return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

df["home_next"] = add_col(df, "home")
df["team_opp_next"] = add_col(df, "team_opp")
df["date_next"] = add_col(df, "date")

full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]], left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])


# In[19]:


removed_columns = list(full.columns[full.dtypes == "object"]) + removed_columns
selected_columns = full.columns[~full.columns.isin(removed_columns)]


# In[20]:


filename = 'rr_model.sav'
if os.path.exists(filename):
    sfs = pickle.load(open(filename, 'rb'))
else:
    sfs.fit(full[selected_columns], full["target"])
    pickle.dump(sfs, open(filename, 'wb'))


# In[21]:


predictors = list(selected_columns[sfs.get_support()])
predictions = backtest(full, rr, predictors, 'target')


# In[22]:


next_game = pd.read_csv('next_game_30.csv',index_col=0)


# In[23]:


# Need to do this before merging into 'Full'
df.team = df.team.str.replace('CHO','CHA').str.replace('PHO','PHX').str.replace('BRK','BKN')
df.team_opp = df.team_opp.str.replace('CHO','CHA').str.replace('PHO','PHX').str.replace('BRK','BKN')
df.team_opp_next = df.team_opp_next.str.replace('CHO','CHA').str.replace('PHO','PHX').str.replace('BRK','BKN')


# In[24]:


next_30 = df[df.target==2]
make_pred = next_30.merge(next_game,how='left',on=['team'])
make_pred.home_next = make_pred.HOME
make_pred.team_opp_next = make_pred.OPPONENT
make_pred.date_next = make_pred.DATE

del make_pred['DATE']
del make_pred['OPPONENT']
del make_pred['HOME']


# In[25]:


rows = df[df['target']==2].index
df.loc[rows,['target','team','team_opp_next','date_next','home_next']] = np.array(make_pred[['target','team','team_opp_next','date_next','home_next']])
full = df.merge(df[rolling_cols + ["team_opp_next", "date_next", "team"]],  left_on=["team", "date_next"], right_on=["team_opp_next", "date_next"])
predictions = backtest(full, rr, predictors,'target')
ml = pd.concat([full,predictions],axis=1)


# In[26]:


today = str(datetime.today())
today = today[0:10]


# In[27]:


ml[(ml.actual==2) & (ml.date_next==today)][['date_next','team_x','team_y','prediction']].sort_values(['date_next','team_x']).sort_values('prediction',ascending=False)


# In[ ]:




