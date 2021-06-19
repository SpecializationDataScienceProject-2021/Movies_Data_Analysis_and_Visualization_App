#required imports
import config
import gzip
import shutil
import pandas as pd
import numpy as np
import requests
import os
import sys
import datetime
from requests_html import HTML

#downloading required file
file_url = "https://datasets.imdbws.com/title.basics.tsv.gz"
r = requests.get(file_url)
with open("title.basics.tsv.gz",'wb') as f:
        f.write(r.content)

#extracting the downloaded file
with gzip.open('title.basics.tsv.gz', 'rb') as f_in:
    with open('title_basics_data.tsv', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

#reading the extracted file
imdbfulldata = pd.read_csv("./title_basics_data.tsv", sep = "\t", low_memory = False)

#removing the compressed file
os.remove("title.basics.tsv.gz")
print("File Removed!")

#Loading the data into a dataframe
df = pd.DataFrame(imdbfulldata)
#Storing only movies data
imdb_movies_data = df[df['titleType'] == 'movie'] 
#Making imdb ids into a single list
imdbids = imdb_movies_data.tconst.tolist() 
print(len(imdbids))

#Getting data from OMDB
omdbresponses = []
for j in imdbids:
    try:
        omdbresponse = requests.get("http://www.omdbapi.com/?i="+ j +"&plot=full&apikey="+config.omdbapikey)
        omdbresponsejsondata = omdbresponse.json()
        print(omdbresponsejsondata)
        if omdbresponsejsondata["Response"] == "True":
            omdbresponses.append(omdbresponsejsondata)
            movies_data = pd.DataFrame(omdbresponses)
            movies_data.to_csv("omdb_movies_data.csv",index=False)
    except:
        pass

#Getting data from TMDB
API_KEY = config.tmdbapikey
df = pd.read_csv("analysed_omdb_movies_data.csv")
imdbids = df["IMDB ID"].tolist()
print(len(imdbids))

tmdbresponsesrevenue = []
tmdbresponsesbudget = []
tmdbresponsesadult = []
tmdbresponsespopularity = []

for i in imdbids:
    print(i)
    print(type(i))
    try:
        tmdbresponse = requests.get("https://api.themoviedb.org/3/movie/"+i+"?api_key="+API_KEY)
        tmdbresponsejson = tmdbresponse.json()
        if tmdbresponse.status_code != 200:
            tmdbresponsesrevenue.append(0)
            tmdbresponsesbudget.append(0)
            tmdbresponsesadult.append("U")
            tmdbresponsespopularity.append(0)
        else:
            tmdbresponsesrevenue.append(tmdbresponsejson["revenue"])
            tmdbresponsesbudget.append(tmdbresponsejson["budget"])
            if tmdbresponsejson["adult"] == "false":
                tmdbresponsesadult.append("U/A")
            else:
                tmdbresponsesadult.append("A")
            tmdbresponsespopularity.append(tmdbresponsejson["popularity"])
    except Exception as e:
        print(e)
        tmdbresponsesrevenue.append(0)
        tmdbresponsesbudget.append(0)
        tmdbresponsesadult.append("U")
        tmdbresponsespopularity.append(0)

df["Revenue"] = tmdbresponsesrevenue
df["Budget"] = tmdbresponsesbudget
df["Popularity"] = tmdbresponsespopularity
df["Ceritificate"]= tmdbresponsesadult
df.to_csv("data_tmdb.csv",index=False)

# Scraping boxoffice mojo data for movie collections
BASE_DIR = os.path.dirname(__file__)

now = datetime.datetime.now()
year = now.year

def url_to_txt(url, filename="world.html", save=False):
    r = requests.get(url)
    if r.status_code == 200:
        html_text = r.text
        if save:
            with open(f"world-{year}.html", 'w') as f:
                f.write(html_text)
        return html_text
    return None

def parse_and_extract(url, name='2020'):
    html_text = url_to_txt(url)
    if html_text is None:
        return False
    r_html = HTML(html=html_text)
    table_class = ".imdb-scroll-table"
    r_table = r_html.find(table_class)
    table_data = []
    table_data_dicts = []
    header_names = []
    if len(r_table) == 0:
        return False
    parsed_table = r_table[0]
    rows = parsed_table.find("tr")
    header_row = rows[0]
    header_cols = header_row.find('th')
    header_names = [x.text for x in header_cols]
    for row in rows[1:]:
        cols = row.find("td")
        row_data = []
        row_dict_data = {}
        for i, col in enumerate(cols):
            row_data.append(col.text)
        table_data_dicts.append(row_dict_data)
        table_data.append(row_data)
    df = pd.DataFrame(table_data, columns=header_names)
    path = os.path.join(BASE_DIR, 'Data')
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join('Data/Years', f'{name}.csv')
    df.to_csv(filepath, index=False)
    return True

def run(start_year=None, years_ago=20):
    if start_year is None:
        now = datetime.datetime.now()
    for i in range(1977,2022):
        url = f"https://www.boxofficemojo.com/year/world/{i}/"
        finished = parse_and_extract(url, name=i)
        if finished:
            print(f"Finished {i}")
        else:
            print(f"{i} not finished")

if __name__ == "__main__":
    try:
        start = int(sys.argv[1])
    except:
        start = None
    try:
        count = int(sys.argv[2])
    except:
        count = 0
    run(start_year=start, years_ago=count)