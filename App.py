# imports
import streamlit as st
import pandas as pd
import numpy as np

# import matplotlib.pyplot as plt
# from wordcloud import WordCloud

import requests

import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from PIL import Image

import warnings
 
icon = Image.open('icon9.jpg')
st.set_page_config(initial_sidebar_state="collapsed", page_title = "MDAV", page_icon = icon)

warnings.simplefilter(action='ignore', category=FutureWarning)

omdburl = 'Data/omdb_movies_data.zip'
analysedomdburl = 'Data/analysed_omdb_movies_data.zip'
tmdburl = 'Data/tmdb_movies_data.csv'
moviesdataurl = 'Data/movies_data.zip'
boxOfficeMojourl = 'Data/boxofficemojo_data.csv'

header = st.beta_container()
image_d = st.beta_container()

# configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

with header:
    st.title('Movies Data Analysis and Visualization')
st.markdown("""In this App we spectate **Movies** data for **Analysis** and **Visualization** to get started click on left top corner **">"** symbol.""")
st.markdown('''* **Python libraries:**streamlit, pandas, numpy, matplotlib, wordcloud, requests, plotly, Image and pyautogui.''')
st.markdown('''* **Data source:** [IMDB](https://datasets.imdbws.com/), [OMDB](https://www.omdbapi.com/), [TMDB](https://developers.themoviedb.org/3/movies/get-movie-details), [BoxOffice Mojo](https://www.boxofficemojo.com/year/world/)''')

with image_d:
    image = Image.open('logo.jpg')
    st.image(image, use_column_width=True)
    
story_select = st.sidebar.selectbox(
    label = "MDAV Options",
    options = ['Select','Analysis','Visualizations']
)

def Analysis():
    st.markdown("""### Analysis of **Movies** data""")
    movies_omdb = pd.read_csv(omdburl, compression='zip', low_memory=False)
    movies_d = pd.read_csv(moviesdataurl, compression='zip', low_memory=False)
    st.markdown("""The **DataTypes**""")
    d1, d2 = st.beta_columns((2,3))
    d1.write(movies_d.dtypes)
    d2.write('The shape of the omdb_movies_data.csv data with duplicates: ')
    d2.write(movies_omdb.shape)
    movies_omdb = movies_omdb.drop_duplicates()
    d2.write('The shape of the omdb_movies_data.csv data after removing duplicates: ')
    d2.write(movies_omdb.shape)
    d2.write('The columns: Title, Year, Realeased, Genre, Director, Actors, Language, Country, Poster, Rating, Votes, IMDB ID, Production, Day, Weekday, Duration, MonthName, Budget, Popularity, Collections, Profit')

    movies_omdb = movies_omdb.rename(columns={'imdbID':'IMDB ID','imdbRating':'Rating','imdbVotes':'Votes'})
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write('For **Released** attribute I have changed the datatype to datetime also got other data attributes like **Month, Monthname, Day, Weekday**')
    movies_omdb['Released']=pd.to_datetime(movies_omdb['Released'])
    rdate = pd.DataFrame()
    rdate['Month'] = movies_omdb['Released'].dt.month 
    rdate['Monthname']= movies_omdb['Released'].dt.strftime("%B")
    rdate['Day'] = movies_omdb['Released'].dt.day
    rdate['Weekday']= movies_omdb['Released'].dt.day_name()
    movies_omdb = movies_omdb.join(rdate)

    st.write('For **Runtime** attribute I have changed the format and changed the column name to **Duration**')
    df2 = movies_omdb['Runtime'].str.split(n=1, expand=True)
    df2.columns = ['Runtime{}'.format(x+1) for x in df2.columns]
    df2 = df2.rename(columns={'Runtime1':"Duration"})
    df2 = df2.drop(columns = ['Runtime2'])
    movies_omdb = movies_omdb.join(df2)
    movies_omdb = movies_omdb.drop(columns = ['Runtime'])
    st.write('Dropped some of unwanted columns like **Rated, Writer, DVD, Plot, Website, Response, Metascore, Ratings, Type, Awards**')
    movies_omdb = movies_omdb.drop(columns=['Rated','Writer','DVD','Plot','Website','Response', 'Metascore', 'Ratings', 'Type', 'Awards'])

    st.write('**Data Cleaning** Handling **Missing Data** removed all null values')
    m1, m2, m3 = st.beta_columns((1, 1, 1))
    m1.write(movies_omdb.isnull().sum())

    # Title
    movies_omdb['Title'] = movies_omdb['Title'].fillna('No Title')

    # Year
    convert_dict = {'Year':int}
    movies_omdb['Year'] = movies_omdb['Year'].fillna(0)
    movies_omdb = movies_omdb.astype(convert_dict)
    
    # Released
    movies_omdb['Released'] = movies_omdb['Released'].fillna('01-01-1800')

    # Genre 
    movies_omdb['Genre'] = movies_omdb['Genre'].fillna('No Genre')

    # Director
    movies_omdb['Director'] = movies_omdb['Director'].fillna('Director Not Mentioned')

    # Actors
    movies_omdb['Actors'] = movies_omdb['Actors'].fillna('Actors Not Mentioned')

    # Lanuguage
    movies_omdb['Language'] = movies_omdb['Language'].fillna('No Language')

    # Country
    movies_omdb['Country'] = movies_omdb['Country'].fillna('No Country')

    # Poster
    movies_omdb['Poster'] = movies_omdb['Poster'].fillna('No Poster')

    # Rating
    movies_omdb['Rating'] = pd.to_numeric(movies_omdb['Rating'],errors='coerce')
    movies_omdb['Rating'] = movies_omdb['Rating'].fillna(0.0)

    # Votes
    movies_omdb['Votes'] = pd.to_numeric(movies_omdb['Votes'],errors='coerce')
    movies_omdb['Votes'] = movies_omdb['Votes'].fillna(0.0)

    # IMDB ID
    movies_omdb['IMDB ID'] = movies_omdb['IMDB ID'].fillna('No imdbID')

    # BoxOffice
    movies_omdb['BoxOffice'] = movies_omdb['BoxOffice'].fillna('No BoxOffice')

    # Production
    movies_omdb['Production'] = movies_omdb['Production'].fillna('No Production')

    # Month
    movies_omdb['Month'] = pd.to_numeric(movies_omdb['Month'],errors='coerce')
    movies_omdb['Month'] = movies_omdb['Month'].fillna(0.0)

    # Monthname
    movies_omdb['MonthName'] = movies_omdb['Monthname'].fillna('No Month')

    # Day
    movies_omdb['Day'] = pd.to_numeric(movies_omdb['Day'],errors='coerce')
    movies_omdb['Day'] = movies_omdb['Day'].fillna(0.0)

    # Weekday
    movies_omdb['Weekday'] = movies_omdb['Weekday'].fillna('No Weekday')

    # Duration
    movies_omdb['Duration'] = pd.to_numeric(movies_omdb['Duration'],errors='coerce')
    movies_omdb['Duration'] = movies_omdb['Duration'].fillna(0)
    movies_omdb['Duration'] = movies_omdb['Duration'].astype(int)

    movies_omdb = movies_omdb.drop(columns=['Monthname', 'Month'])

    m2.write(movies_omdb.isnull().sum())

    # Appending all files from different API's
    df1 = pd.read_csv(analysedomdburl, compression='zip', low_memory=False)
    df2 = df1.drop_duplicates()
    df3 = df2.drop_duplicates(subset=["IMDB ID"])
    df4 = pd.read_csv(boxOfficeMojourl)
    df5 = df4.drop_duplicates()
    df6 = pd.read_csv(tmdburl)
    df7 = df6.drop_duplicates()
    df = pd.DataFrame()
    df = df3.merge(df5, on='Title', how="left")
    df = df.merge(df7, on='IMDB ID', how="left")

    st.write('**Inflation rate** converting **USD** to **INR** and also added **Profit** Column for visualization purpose')
    # getting the conversion value of usd to inr
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    data = requests.get(url).json()
    currencies = data['rates']
    indiancurrency = currencies['INR']

    df["BoxOffice"] = df["BoxOffice"].str.replace(r'\D', '')
    df['BoxOffice'] = df['BoxOffice'].replace(np.NaN, 0)
    df["Worldwide"] = df["Worldwide"].str.replace(r'\D', '')
    df['Worldwide'] = df['Worldwide'].replace(np.NaN, 0)
    df["BoxOffice"] = pd.to_numeric(df['BoxOffice'],errors='coerce')
    df["Worldwide"] = pd.to_numeric(df['Worldwide'],errors='coerce')

    #Inflation is done here
    df["Budget"] = df["Budget"] *indiancurrency
    df["Collections"] = (df['BoxOffice']*indiancurrency)+(df['Revenue']*indiancurrency)+(df['Worldwide']*indiancurrency)
    df = df.drop(columns=['BoxOffice','Revenue','Worldwide'])
    df = df.drop_duplicates(subset=['IMDB ID'])

    df['Profit'] = df['Collections'] - df['Budget']

    # Budget
    df['Budget'] = df['Budget'].fillna(0.0)

    # Popularity
    df['Popularity'] = df['Popularity'].fillna(0.0)

    # Collections
    df['Collections'] = df['Collections'].fillna(0.0)

    # Profit
    df['Profit'] = df['Profit'].fillna(0.0)

    df = df.drop(columns=['Certificate'])

    df_c = df.loc[df['Collections'] > 0]
    st.write(df_c.head())

    m3.write(df.isnull().sum())

    movies_data = pd.read_csv(moviesdataurl, compression = 'zip', low_memory=False) 

    st.markdown("""The **first** five rows data:""")
    st.write(movies_data.head(5))
    st.markdown("""The **last** five rows data:""")
    st.write(movies_data.tail(5))

    st.write('**Statistical** data after data cleaning')
    st.write(movies_data.describe())

# displayed analysis part by clicking option
if story_select == 'Analysis':
    Analysis()

# Application options
def MDAV_options():
    option = st.sidebar.radio('Visualization options', ['Worldwide', 'Indian', 'USA'])
    movies_data = pd.read_csv(moviesdataurl, compression = 'zip', low_memory=False) 
    if option == 'Worldwide':
        st.write('**Worldwide** visualizations')
        df = movies_data

    if option == 'Indian':
        st.write('**Indian** visualizations')
        df = movies_data
        df = df[df['Country'] == 'India']

    if option == 'USA':
        st.write('**United States of America** visualizations')
        df = movies_data
        df = df[df['Country'] == 'USA']

    story_select = st.sidebar.selectbox(
    label = "Select the story to visualise",
    options = ['Select','Numberof_Movies_by_gcl', 'Year_vs_Movies', 'Max_BoxOffice_Movies_each_Year', 'Ratings_distribution', 
    'Maximum_Rated_Movies', 'Movies_based_datecount', 'Top_10_Movies_Titles','PieChart_noof_movies_by_Year','Word_visualizations',
    'Genres_of_2000s_movies','Statistical_BoxOffice_by_Years','Duration_distribuion','Differentiation_scatters', 
    'Crew_movies_count','All Stories'])

    def Numberof_Movies_by_gcl():
        try:
            
            st.info("""We have visualized Between **Year** and **Genres**, **Country** and **Language** count from 1990 to 2021. please 
            select the dropdown options to visualize graphs""")
            story_select_gcl = st.selectbox(
            label = "Try all options",
            options = ['Select','Genre', 'Country', 'Language'])

            if story_select_gcl == "Genre":
                st.success("""**Data:** Year, Genre, **Why:** to show the change over time so we selected **linePlot**""")

                st.info("""We have visualized Between **Year** and top **Genres** count from 1990 to 2021 According 
                to the visualization for **WorldWide** movies the **Documentary** is more. Whereas for **Indian** movies 
                **Drama** is more and for **USA** movies the **Documentary** and **Drama** are equally overlaped at each 
                other point""")
                slider_range_year = st.slider("Select the range of year", 1990, 2021 , (1990, 2021))
                df_genres_sub = df[df['Year'] > slider_range_year[0]]
                df_genres_sub = df_genres_sub[df_genres_sub['Year'] < slider_range_year[1]]

                df_series = (df_genres_sub.groupby(['Genre', 'Year']).size())
                df_series = df_series.unstack(level=0)
                df_series['sum'] = df_series.sum(axis = 1)
                df_top_10 = df_series.loc[:, ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Documentary', 'Biography', 'Adventure', 'Crime']]
                df_top = df_top_10.reset_index()
                df_top.head()
                fig1 = px.line(df_top, 
                    x = 'Year', 
                    y = ['Drama', 'Comedy', 'Thriller', 'Romance', 'Action', 'Horror', 'Documentary', 'Biography', 'Adventure', 'Crime'], 
                    title = 'Count of the 10 most popular genres produced from 1990 to 2021')
                st.plotly_chart(fig1)

                st.success("""**Data:** Genre, Year, **Why: ** Compare values between groups so we selected 
                **Bar plot**, groups: Genres, Comparision: Year """)
                st.info("""We have visualized the top **Genres** of all **Years** from **1900 to 2021** and their count. For **Worldwide** 
                and **Indian** movies its **Drama** whereas for **USA** its **Drama & Documentary**""")
                df_genre = df[df['Genre'] != 'No Genre']
                df_genre = df_genre[['Genre', 'Year']]
                cnt = df_genre.groupby(['Genre']).count().reset_index()
                cnt.rename(columns = {'Year':'Count'},inplace = True)
                scnt = cnt.sort_values('Count', ascending = False)
                df_count = scnt.head(10)
                fig4 = px.bar(df_count, 
                    x = 'Count', 
                    y = 'Genre', 
                    orientation = 'h', 
                    width = 800, 
                    title = 'Top 10 Genres of Movies Produced from 1900 to 2021')
                st.plotly_chart(fig4)

            if story_select_gcl == "Country":
                st.success("""**Data:** Country, Title, **Why: ** Compare values between groups so we selected 
                **Bar plot**, groups: Country, Comparision: Number of movies """)
                st.info("""We have visualized The **Country** based count of movies According to the visualization the **Worldwide** movies 
                were produced more by The country **USA** with count of **113.906K** and for **Indian** movies the count is **33.358K**""")
                df_country_list = df['Country'].to_list()
                s = ','.join(df_country_list)
                listt = s.split(",")
                freq = {} 
                for item in listt: 
                    if (item in freq): 
                        freq[item] += 1
                    else: 
                        freq[item] = 1

                df_actors_l_f = pd.DataFrame(list(freq.items()),columns = ['Country','No_of_movies'])
                scnt = df_actors_l_f.sort_values(by=['No_of_movies'])
                fig10 = px.bar(scnt.tail(30), x="Country", y="No_of_movies", title='Country Vs Number Of Movies', color="Country")
                st.plotly_chart(fig10)

            if story_select_gcl == "Language":
                st.success("""**Data:** Language, IMDB ID, **Why: ** Compare values between groups so we selected 
                **Bar plot**, groups: Language, Comparision: Number of movies """)
                st.info("""We have visualized The known Languages based on the story **Word_visualizations**. In this The most released movies 
                **Language** according to **WorldWide** and **USA** movies is **English**. Whereas for **Indian** movies is **Hindi**""")
                df2 = df.groupby("Language").agg({"IMDB ID": pd.Series.nunique}).reset_index()
                df2.rename(columns = {'IMDB ID':'Number of Movies'},inplace = True)
                data = df2.drop([9], axis=0)
                data = data.loc[data['Language'].isin(['Telugu','Hindi','English','Tamil', 'Kannada', 'Malayalam', 'Urdu', 'Spanish', 'French', 'Marathi', 'Japanese', 'German'])]
                data1 = data[data["Number of Movies"] > 25]
                fig8 = px.bar(data1, x = 'Language', y = 'Number of Movies', title = 'Number of movies by Language')
                fig8.update_traces(textposition='outside')
                fig8.update_layout(uniformtext_minsize=8)
                st.plotly_chart(fig8)

        except Exception as e:
            print(e)

    def Year_vs_Movies():
        try:
            st.success("""**Data:** Year, IMDB ID, **Why:** to show the change over time so we selected **linePlot**, comparision: number of movies and 
            time: years""")
            st.info("""We have visualized Between **Year** wise **IMDB ID** count from 1897 to 2020 According to 
            the visualization of **WorldWide** movies, **Indian** movies and **USA** movies **2018** year movies 
            count is more""")
            slider_range_year = st.slider("Select the range of year", 1897, 2025 , (1990, 2024))
            df_s = df[df['Year'] > slider_range_year[0]]
            df_s = df_s[df_s['Year'] < slider_range_year[1]]
            data=df_s.groupby('Year').count()['IMDB ID'].reset_index()
            data.rename(columns = {'IMDB ID':'Number Of Movies'},inplace = True)
            fig2 = px.line(data, x="Year", y="Number Of Movies", title='Year Vs Number Of Movies')
            st.plotly_chart(fig2)
        except Exception as e:
            print(e)

    def Max_BoxOffice_Movies_each_Year():
        try:
            st.success("""**Data: ** Year, Collections, Budget, Profit, **Why: ** Compare values between groups so we selected 
            **Bar plot**, groups: Year, Comparision: BoxOffice data""")
            st.info("""We have visualized the Maximum **BoxOffice** data in each year of 2000's based on three 
            attributes **Collections**, **Budget**, **Profit** the table below are some of the observations, The 
            **WorldWide** and **USA** movies data are having similar visualizations compared to **Indian** movies data.
            please select boxOffice option from dropdown to visualize""")
            st.markdown('''|         |Year      |Collections|Year|Budget|Year|Profit| 
                        |-------- |----------|-----------|----|------|----|------|
                        |WorldWide|2009, 2019|15.2T      |2011|2.03T |2009|13.9T |
                        |Indian   |2017      |1.5T       |2018|412B  |2017|1.33T |
                        |USA      |2009, 2019|15.2T      |2019|1.91T |2009|13.9T |''')
            story_select = st.selectbox(
            label = "Try all options",
            options = ['Select','Collections', 'Budget', 'Profit'])
            if story_select == "Collections":
                small_df = df[['Title', 'Year', 'Collections']]
                small_df = df[df['Year'] > 2000]
                small_df = small_df[small_df['Year'] < 2021]
                df1 = small_df.sort_values(['Year', 'Collections'], ascending = [True, False])
                df2 = df1.groupby('Year').apply(lambda x: x[x['Collections'] == x['Collections'].max()])
                fig3 = px.bar(df2, x = 'Year', y = 'Collections', text = 'Title', title ='Maximum Collections of movie in each Year')
                st.plotly_chart(fig3)
            if story_select == "Budget":
                small_df = df[['Title', 'Year', 'Budget']]
                small_df = df[df['Year'] > 2000]
                small_df = small_df[small_df['Year'] < 2021]
                df1 = small_df.sort_values(['Year', 'Budget'], ascending = [True, False])
                df2 = df1.groupby('Year').apply(lambda x: x[x['Budget'] == x['Budget'].max()])
                fig3 = px.bar(df2, x = 'Year', y = 'Budget', text = 'Title', title ='Maximum Budgets of movie in each Year')
                st.plotly_chart(fig3)
            if story_select == "Profit":
                small_df = df[['Title', 'Year', 'Profit']]
                small_df = df[df['Year'] > 2000]
                small_df = small_df[small_df['Year'] < 2021]
                df1 = small_df.sort_values(['Year', 'Profit'], ascending = [True, False])
                df2 = df1.groupby('Year').apply(lambda x: x[x['Profit'] == x['Profit'].max()])
                fig3 = px.bar(df2, x = 'Year', y = 'Profit', text = 'Title', title ='Maximum Profits of movie in each Year')
                st.plotly_chart(fig3)
        except Exception as e:
            print(e)

    def Ratings_distribution():
        try:
            st.success("""**Data: ** Rating, Title **Why: ** Distribution of numeric variable values as a series of bar so 
            we selected **Histogram plot**, numeric variable: Rating""")
            st.info("""We have visualized the **Rating** of all movies and their count. For **Worldwide** the distribution 
            occurs at **6.4** with count **6802**. whereas for **Indian** movies the distribution occurs at **7.2** with count 
            of **466**. and for **USA** the distribution occurs at **6.4** with count **1982**""")
            cnt = df.groupby(['Rating'])['Title'].count().reset_index()
            cnt.rename(columns = {'Title':'Number of Movies'},inplace = True)
            scnt = cnt.sort_values('Number of Movies', ascending = False)
            scnt = scnt.iloc[1: , :]
            fig7 = px.bar(scnt, x="Rating", y="Number of Movies", title='Rating Vs Number Of Movies', color="Rating")
            st.plotly_chart(fig7)
        except Exception as e:
            print(e)

    def Maximum_Rated_Movies():
        try:
            st.success("""**Data: ** Year, Rating, Title **Why: ** Compared values between groups so we selected 
            **Bar plot**, groups: Year, Comparision: Rating""")
            st.info("""We have visualized the Maximum **Rating** movies in each year from **1900 to 2020**. 
            For this visualization we have included slider value. For **World Wide**, **Indian** and **USA** 
            movies we can observe so many **10** rated movies please adjust the slider to know more in detail""")
            slider_range_mam = st.slider("Select the range to visualize", 1900, 2020 , (1900, 2020))
            small_df = df[['Title', 'Year', 'Rating']]
            small_df = small_df[small_df['Year'] > slider_range_mam[0]]
            small_df = small_df[small_df['Year'] < slider_range_mam[1]]
            df1 = small_df.sort_values(['Year', 'Rating'], ascending = [True, False])
            df2 = df1.groupby('Year').apply(lambda x: x[x['Rating'] == x['Rating'].max()])
            fig8 = px.bar(df2, x = 'Year', y = 'Rating', text = 'Title', title = 'Maximum Rated movies from 1900 to 2020', color="Year")
            fig8.update_traces(textposition='outside')
            fig8.update_layout(uniformtext_minsize=8)
            st.plotly_chart(fig8)
        except Exception as e:
            print(e)

    def Movies_based_datecount():
        try:
            st.success("""**Data: ** Day, Weekday, Month, Year, IMDB ID **Why: ** Compared values between groups so we selected 
            **Bar plot**, groups: Day, Weekday, Month, Comparision: Number of movies""")
            st.info("""We have visualized between the **Year** and the count of **Released Year** attributes like **Day wise**, 
            **WeekDay wise** and **Month wise**. The highest count for **WorldWide** movies the **date** is **1**, **Weekday** is 
            **Friday** and **Month name** is **October** Whereas, for **Indian** movies the **date** is **1**, **Weekday** is 
            **Friday** and **Month name** is **January** and for **USA** movies the **date** is **1**, **Weekday** is **Friday** and 
            **Month name** is **October**,""")
            story_select = st.selectbox(
            label = "Try all options",
            options = ['Select','Date', 'Weekday', 'Month'])
            if story_select == 'Date':
                df2 = df[df['Year'] > 1997]
                df2 = df.loc[df["Day"] != 0]
                df2 = df2.groupby("Day").agg({"IMDB ID": pd.Series.nunique}).reset_index()
                df2.rename(columns = {'IMDB ID':'Number of Movies'},inplace = True)
                data = df2.drop([2], axis=0)
                fig8 = px.bar(data, x = 'Day', y = 'Number of Movies', title = 'Number of movies by Date')
                fig8.update_traces(textposition='outside')
                fig8.update_layout(uniformtext_minsize=8)
                st.plotly_chart(fig8)

            if story_select == 'Weekday':
                df2 = df.groupby("Weekday").agg({"IMDB ID": pd.Series.nunique}).reset_index()
                df2.rename(columns = {'IMDB ID':'Number of Movies'},inplace = True)
                data = df2.drop([2], axis=0)
                fig8 = px.bar(data, x = 'Weekday', y = 'Number of Movies', title = 'Number of movies by Weekday')
                fig8.update_traces(textposition='outside')
                fig8.update_layout(uniformtext_minsize=8)
                st.plotly_chart(fig8)
            
            if story_select == 'Month':
                df2 = df.groupby("MonthName").agg({"IMDB ID": pd.Series.nunique}).reset_index()
                df2.rename(columns = {'IMDB ID':'Number of Movies'},inplace = True)
                data = df2.drop([9], axis=0)
                fig8 = px.bar(data, x = 'MonthName', y = 'Number of Movies', title = 'Number of movies by MonthName')
                fig8.update_traces(textposition='outside')
                fig8.update_layout(uniformtext_minsize=8)
                st.plotly_chart(fig8)
        except Exception as e:
            print(e)

    def Top_10_Movies_Titles():
        try:
            st.success("""**Data:** Budget, Title, Why: To show the differentiation of top 10 BoxOffice data as well as the longest popular
            movies data so we selected lines+marker graph using scatter plot""")

            st.info("""We have visualized Top most 10 movies based on **BoxOffice** data which consists of
            **Budget**, **Collections** and **Profit** and as well as We have visualized The Top 10 longest movies based on 
            **Duration** and Popular movies based on **Popularity**. Please select the dropdown to visualize the visualizations""")
            # For **WorldWide** as well as **USA** 
            # movies the longest duration movie was **Welcome to NewYork** with Duration **950** and for 
            # **Indian** movies the longest movie was **CzecMate: In Search of Jiri Menzel** with Duration 
            # **426**. comming to **Popularity** for **WorldWide** and **USA** movies the popular movie was 
            # **Tom Clancy's Without Remorse** with popularity count **4000K**. For **Indian** movies the 
            # popular movie was **Drive** with popularity count **91K** the below table shows all the top 10 movies data 
            # try selecting all the three attributes.
            story_select = st.selectbox(
            label = "Try all options",
            options = ['Select','Budget', 'Collections', 'Profit', 'Duration', 'Popularity'])

            if story_select == "Budget":
                df8 = pd.DataFrame(df['Budget'].sort_values(ascending = False))
                df8['Title'] = df['Title']
                data = list(map(str,(df8['Title'])))
                # extract the top 10 budget movies data from the list and dataframe.
                x = list(data[1:11])
                y = list(df8['Budget'][1:11])
                #plot the figure and setup the title and labels.
                fig = go.Figure(data = go.Scatter(x=y, y=x, mode='lines+markers'))
                fig.update_layout(title='Budget based Top movies')
                st.plotly_chart(fig)

            if story_select == "Collections":
                df8 = pd.DataFrame(df['Collections'].sort_values(ascending = False))
                df8['Title'] = df['Title']
                data = list(map(str,(df8['Title'])))
                # extract the top 10 Collection movies data from the list and dataframe.
                x = list(data[1:11])
                y = list(df8['Collections'][1:11])
                #plot the figure and setup the title and labels.
                fig = go.Figure(data = go.Scatter(x=y, y=x, mode='lines+markers'))
                fig.update_layout(title='Profit based Top movies')
                st.plotly_chart(fig)

            if story_select == "Profit":
                df8 = pd.DataFrame(df['Profit'].sort_values(ascending = False))
                df8['Title'] = df['Title']
                data = list(map(str,(df8['Title'])))
                # extract the top 10 Profit movies data from the list and dataframe.
                x = list(data[1:11])
                y = list(df8['Profit'][1:11])
                #plot the figure and setup the title and labels.
                fig = go.Figure(data = go.Scatter(x=y, y=x, mode='lines+markers'))
                fig.update_layout(title='Profit based Top movies')
                st.plotly_chart(fig)

            if story_select == "Duration":
                df_d = pd.DataFrame(df['Duration'].sort_values(ascending = False))
                df_d['Title'] = df['Title']
                data = list(map(str,(df_d['Title'])))
                #extract the top 10 longest duraton movies data from the list and dataframe.
                x = list(data[:10])
                y = list(df_d['Duration'][:10])
                fig = go.Figure(data = go.Scatter(x=y, y=x, mode='lines+markers'))
                fig.update_layout(title='Top 10 Longest Duration Movies')
                st.plotly_chart(fig)

            if story_select == "Popularity":
                df_p = pd.DataFrame(df['Popularity'].sort_values(ascending = False))
                df_p['Title'] = df['Title']
                data = list(map(str,(df_p['Title'])))
                #extract the top 10 longest duraton movies data from the list and dataframe.
                x = list(data[:10])
                y = list(df_p['Popularity'][:10])
                fig1 = go.Figure(data = go.Scatter(x=y, y=x, mode='lines+markers'))
                fig1.update_layout(title='Top 10 Popular Movies')
                st.plotly_chart(fig1)
        except Exception as e:
            print(e)

    def PieChart_noof_movies_by_Year():
        try:
            st.success("""**Data: ** Year, Title, **Why: ** to display the composition of categorical data category: Year""")
            st.info("""We have visualized The **Donut PieChart** between **Year** and **Title**. Here we have shown percentage of each year count of movies.
            Included the slider for selecting Years range. To know more please explore the slider""")
            slider_range = st.slider("Select the range to visualize", 1990, 2021 , (1990, 2021))
            movies_countries_df = df[df['Year'] > slider_range[0]]
            movies_countries_df = movies_countries_df[movies_countries_df['Year'] <= slider_range[1]]
            movies_countries_df = movies_countries_df[['Year','Title']].groupby(['Year']).count().reset_index().rename(columns={'Title':'number_of_movies'})
            movies_countries_df = movies_countries_df.sort_values(by=['Year'], ascending=False)
            fig = px.pie(movies_countries_df, values='number_of_movies', names='Year', title='Title based number of movies by Year', width = 600, height=550)
            fig.update_traces(hoverinfo='label+percent', hole=.2)
            st.plotly_chart(fig)
        except Exception as e:
            print(e)

    def Word_visualizations():
        try:
            st.success("""**Data: ** Genre, Title, Language, Country, **Why: ** To show how frequently word appears in the given attributes
            so we selected wordCloud """)
            st.info("""We have visualized The **WordCloud** for **Genres**, **Title** and **Language** to check the more reused 
            words from all the data according to the given attributes below table shows some of observation""")
            st.markdown('''|         |Genre               |Title                |Language|Country   | 
                        |---------|--------------------|---------------------|--------|----------|
                        |WorldWide|Drama, Comedy       |Love, Life, de, Man  |English |USA, Japan|
                        |Indian   |Drama               |Love, Ki, Ek         |Hindi   |India     |
                        |USA      |Drama, Comedy, Crime|Love, Life, Man, Girl|English |USA       |''')
            story_select = st.selectbox(
            label = "Try all options",
            options = ['Select', 'Genre', 'Title', 'Language', 'Country'])
            if story_select == "Genre" or "Select":
                st.set_option('deprecation.showPyplotGlobalUse', False)
                if option == "Worldwide":
                    wordcloudimage = Image.open('WordCloud/wordcloud_genre_world.jpg')
                    st.image(wordcloudimage)
                elif option == "Indian":
                    wordcloudimage = Image.open('WordCloud/wordcloud_genre_indian.jpg')
                    st.image(wordcloudimage)
                else:
                    wordcloudimage = Image.open('WordCloud/wordcloud_genre_usa.jpg')
                    st.image(wordcloudimage)
            if story_select == "Title":
                if option == "Worldwide":
                    wordcloudimage = Image.open('WordCloud/wordcloud_title_world.jpg')
                    st.image(wordcloudimage)
                elif option == "Indian":
                    wordcloudimage = Image.open('WordCloud/wordcloud_title_indian.jpg')
                    st.image(wordcloudimage)
                else:
                    wordcloudimage = Image.open('WordCloud/wordcloud_title_usa.jpg')
                    st.image(wordcloudimage)
            if story_select == "Language":
                if option == "Worldwide":
                    wordcloudimage = Image.open('WordCloud/wordcloud_lang_world.jpg')
                    st.image(wordcloudimage)
                elif option == "Indian":
                    wordcloudimage = Image.open('WordCloud/wordcloud_lang_indian.jpg')
                    st.image(wordcloudimage)
                else:
                    wordcloudimage = Image.open('WordCloud/wordcloud_lang_usa.jpg')
                    st.image(wordcloudimage)
            if story_select == "Country":
                if option == "Worldwide":
                    wordcloudimage = Image.open('WordCloud/wordcloud_country_world.jpg')
                    st.image(wordcloudimage)
                elif option == "Indian":
                    wordcloudimage = Image.open('WordCloud/wordcloud_country_indian.jpg')
                    st.image(wordcloudimage)
                else:
                    wordcloudimage = Image.open('WordCloud/wordcloud_country_usa.jpg')
                    st.image(wordcloudimage)
        except Exception as e:
            print(e)

    def Genres_of_2000s_movies():
        try:
            st.success("""**Data: ** Year, Title, Genre, **Why: ** Compared values between groups so we selected 
            **Bar plot**, groups: Year, Genre Comparision: Number of movies""")
            st.info("""We have visualized the **Genres** and **Years** of **2000's**. In this classification for **Worldwide** and **USA** movies 
            we have more **Documentary** for all  2000s Years whereas for **Indian** movies we have **Drama** for all 2000s Yaers. To know more 
            please use slider.""")
            slider_range = st.slider("Select the range to visualize", 2000, 2021 , (2000, 2021))
            years_df = df[['Year','Title', 'Genre']].groupby(['Year','Genre']).count().reset_index().rename(columns={'Title':'number_of_movies'})
            years_df = years_df.sort_values(by=['number_of_movies'], ascending=False)
            years_df = years_df[years_df['Year'] >= slider_range[0]]
            years_df = years_df[years_df['Year'] < slider_range[1]]

            years_df = years_df.loc[years_df["Genre"] != 'No Genre']
            fig11 = px.bar(years_df.head(100), 
                                    x ='Year', 
                                    y = 'number_of_movies', 
                                    color = 'Genre', 
                                    title='Movies produced in the 2000s classified by genres', 
                                    text = 'number_of_movies',
                                    labels = dict(Year = 'year', number_of_movies = 'Number of movies')
                                    )
            st.plotly_chart(fig11)
        except Exception as e:
            print(e)

    def Statistical_BoxOffice_by_Years():
        try:
            st.success("""**Data: **Budget, Collections, Profit, Year, **Why:** we are showing the relation between BoxOffice and Average BoxOffice
            over years""")
            st.info("""We have visualized The Boxoffice **Statistical** data which are **Mean and Maximum** implemented on
            **Budget, Collections, and Profit** for years **1900** to **2021**""")
            Boxoffice_select = st.selectbox(
            label = "Try all options",
            options = ['Select', 'Average', 'Highest'])
            slider_range_box = st.slider("Select the range of year", 1900, 2021 , (1900, 2021))
            data = df[df['Year']>slider_range_box[0]]
            data = data[data['Year'] <= slider_range_box[1]]
            data = data.loc[data["Budget"] > 0]
            data = data.loc[data["Collections"] > 0]
            data = data.loc[data["Profit"] > 0]
            if Boxoffice_select == "Average":
                data1 = data.groupby('Year').mean()['Budget'].reset_index()
                fig1 = px.scatter(data1, x="Year", y="Budget", title='The Average Budget of years 1900-2021', color="Year")
                st.plotly_chart(fig1)

                data2 = data.groupby('Year').mean()['Collections'].reset_index()
                fig2 = px.scatter(data2, x="Year", y="Collections", title='The Average of Collections for years 1900-2021', color="Year")
                st.plotly_chart(fig2)

                data3 = data.groupby('Year').mean()['Profit'].reset_index()
                fig3 = px.scatter(data3, x="Year", y="Profit", title='The Average Profit for years 1900-2021', color="Year")
                st.plotly_chart(fig3)

            if Boxoffice_select == "Highest":
                data1 = data.groupby('Year').max()['Budget'].reset_index()
                fig1 = px.scatter(data1, x="Year", y="Budget", title='The Maximum Budget of years 1990-2021', color="Year")
                st.plotly_chart(fig1)

                data2 = data.groupby('Year').max()['Collections'].reset_index()
                fig2 = px.scatter(data2, x="Year", y="Collections", title='The Maximum of Collections for years 1900-2021', color="Year")
                st.plotly_chart(fig2)

                data3 = data.groupby('Year').max()['Profit'].reset_index()
                fig3 = px.scatter(data3, x="Year", y="Profit", title='The Maximum Profit for years 1900-2021', color="Year")
                st.plotly_chart(fig3)
        except Exception as e:
            print(e)

    def Duration_distribuion():
        try:
            st.info("""We have visualized The **Duration** Mean Distribution for **Rating**. For **WorldWide** and **USA** movies the highest distribution occurs at
            **4.6** rating. For **Indian** movies the distribution occurs at **3.8** rating.""")
            data = df.groupby('Duration')['Rating'].mean().reset_index()
            x = data["Rating"]
            hist_data = [x]
            group_labels = ['Duration']
            fig = ff.create_distplot(hist_data, group_labels)
            fig.update_layout(title_text='Mean Distribution of Duration and Rating')
            st.plotly_chart(fig)

            st.info("""These two visualizations states that the **count** of each **Duration** by using Slider. 
            Some of observations are for **WorldWide** and for **USA** movies the highest count of duration is 
            **90**. Whereas for **Indian** movies the highest duration count is **120**""")
            df2 = df[df['Year'] <= 2021]
            slider_range_d = st.slider("Select the range to visualize", 10, 1000 , (60, 130))
            df2 = df2.loc[df["Duration"] != 0]
            df2 = df2[df2['Duration'] > slider_range_d[0]]
            df2 = df2[df2['Duration'] < slider_range_d[1]]
            df2 = df2.groupby("Duration").agg({"IMDB ID": pd.Series.nunique}).reset_index()
            df2.rename(columns = {'IMDB ID':'NumberofMovies'},inplace = True)
            # data = df2["Duration"]

            fig8 = px.bar(df2, x = 'Duration', y = 'NumberofMovies', title = 'Number of movies by Duration')
            fig8.update_traces(textposition='outside')
            fig8.update_layout(uniformtext_minsize=8)
            st.plotly_chart(fig8)

            fig3 = go.Figure(data=[go.Scatter(
                x = df2.Duration,
                y = df2["NumberofMovies"],
                mode='markers',marker=dict(
                color=['rgb(93, 164, 214)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)'])
            )])
            st.plotly_chart(fig3)

        except Exception as e:
            print(e)

    def Differentiation_scatters():
        try:
            st.info("""We have visualized The scatter plots of 100 data for three attributes **Popularity, Duration and Votes**. 
            In this visualization we can observe common scatters between three scatter plots. """)
            df1 = df.head(100).copy()
            trace1 =go.Scatter(
                x =df1.index,
                y = df1.Popularity,
                mode ="lines",
                name = " Popularity",
                marker = dict(color = "rgb(242, 99, 74,0.7)"),
                text = df1.Title
            )
            trace2 = go.Scatter(
                x = df1.index,
                y = df1.Duration,
                mode = "lines + markers",
                name = "Duration",
                marker = dict( color = "rgb(144, 211, 74,0.5)"),
                text = df1.Title
            )
            trace3 = go.Scatter(
                x = df1.index,
                y = df1.Votes,
                mode = "markers",
                name = "Votes",
                marker = dict(color = "rgb(118, 144, 165)"),
                text = df1.Title
            )
            data1 = [trace1]
            layout = dict(
                title = "Popularity"
            )
            fig = dict ( data = data1 , layout = layout)
            st.plotly_chart(fig)

            data2 = [trace2]
            layout = dict(
                title = "Duration"
            )
            fig1 = dict ( data = data2 , layout = layout)
            st.plotly_chart(fig1)

            data3 = [trace3]
            layout = dict(
                title = "Votes"
            )
            fig3 = dict ( data = data3 , layout = layout)
            st.plotly_chart(fig3)
        except Exception as e:
            print(e)

    def Crew_movies_count():
        try:
            st.info("""We have visualized The 20 known **Directors**, **Actors** and **Actress**, movies count 
            based on the frequency for **WorldWide** and **USA** movies data the highest movies director is **William Beaudine** and for 
            **Indian** movies data the highest movies director is **Narayana Rao Dasari**. For **WorldWide** and **Indian** movies data 
            the highest movies actor is **Chiranjeevi** and actress is **Sridevi**, For **USA** movies the highest movie actor is 
            **Mickey Rooney** and actress is **Katharine Hepburn**""")
            story_select = st.selectbox(
            label = "Try all options",
            options = ['Select', 'Actors', 'Actress', 'Director'])
            if story_select == 'Actors' or 'Select':
                data = df.loc[df["Actors"] != 'Actors Not Mentioned']
                actors = []
                for i in data['Actors']:
                    a = i.split(", ")
                    for j in a:
                        actors.append(j)

                frequencies = {}
                for item in actors:
                    if item in frequencies:
                        frequencies[item] += 1
                    else:
                        frequencies[item] = 1

                df11 = pd.DataFrame()
                df11["Actor"] = frequencies.keys()
                df11["Number_of_Movies"] = frequencies.values()
                df1 = df11.loc[df11['Actor'].isin(['Leonardo DiCaprio','John Carradine','Mickey Rooney','Vikram', 'Chiranjeevi','Nagarjuna Akkineni',
                'Akshay Kumar','Amitabh Bachchan','Shah Rukh Khan','Venkatesh Daggubati','Nandamuri Balakrishna', 'Pawan Kalyan', 'Mahesh Babu','Nani', 
                'Will Smith', 'Johnny Depp','Prabhas', 'Vijay Sethupathi', 'Dulquer Salmaan', 'Rajinikanth'])]
                fig = px.line(df1, x="Actor", y="Number_of_Movies", title='Actors and their Movies Count') 
                st.plotly_chart(fig)
            if story_select == 'Actress':
                data = df.loc[df["Actors"] != 'Actors Not Mentioned']
                actors = []
                for i in data['Actors']:
                    a = i.split(", ")
                    for j in a:
                        actors.append(j)

                frequencies = {}
                for item in actors:
                    if item in frequencies:
                        frequencies[item] += 1
                    else:
                        frequencies[item] = 1

                df11 = pd.DataFrame()
                df11["Actor"] = frequencies.keys()
                df11["Number_of_Movies"] = frequencies.values()
                df2 = df11.loc[df11['Actor'].isin(['Kajal Aggarwal', 'Anushka Shetty', 'Tamannaah Bhatia', 'Nayanthara', 'Scarlett Johansson', 'Angelina Jolie', 
                'Deepika Padukone', 'Raashi Khanna', 'Priyanka Chopra', 'Savitri', 'Jamuna', 'Simran', 'Payal Rajput', 'Katharine Hepburn', 'Shruti Haasan', 
                'Nazriya Nazim', 'Margot Robbie', 'Samantha Akkineni', 'Hema Malini', 'Sridevi'])]
                df2.rename(columns = {'Actor':'Actress'},inplace = True)
                fig2 = px.line(df2, x="Actress", y="Number_of_Movies", title='Actress and their Movies Count')
                st.plotly_chart(fig2)
            if story_select == 'Director':
                data = df.loc[df["Director"] != 'Director Not Mentioned']
                actors = []
                for i in data['Director']:
                    a = i.split(", ")
                    for j in a:
                        actors.append(j)

                frequencies = {}
                for item in actors:
                    if item in frequencies:
                        frequencies[item] += 1
                    else:
                        frequencies[item] = 1

                df11 = pd.DataFrame()
                df11["Directors"] = frequencies.keys()
                df11["Number_of_Movies"] = frequencies.values()
                df1 = df11.loc[df11['Directors'].isin(['S.S. Rajamouli','Trivikram Srinivas','K. Raghavendra Rao','Bapu', 'Narayana Rao Dasari',
                'Singeetham Srinivasa Rao','Jandhyala','Rohit Shetty','David Dhawan','Mahesh Bhatt','Sasikumar', 'Mani Ratnam', 'S. Shankar'
                ,'Sathyan Anthikad', 'K.S. Sethumadhavan', 'T. Hariharan','Richard Thorpe', 'James Cameron', 'Louis Feuillade', 'William Beaudine'])]
                fig = px.line(df1, x="Directors", y="Number_of_Movies", title='Director and their Movies Count')
                st.plotly_chart(fig)    
        except Exception as e:
            print(e)

    if story_select == 'Numberof_Movies_by_gcl':
        Numberof_Movies_by_gcl()

    if story_select == 'Year_vs_Movies':
        Year_vs_Movies()

    if story_select == 'Max_BoxOffice_Movies_each_Year':
        Max_BoxOffice_Movies_each_Year()

    if story_select == 'Ratings_distribution':
        Ratings_distribution()

    if story_select == 'Maximum_Rated_Movies':
        Maximum_Rated_Movies()

    if story_select == 'Movies_based_datecount':
        Movies_based_datecount()

    if story_select == 'Top_10_Movies_Titles':
        Top_10_Movies_Titles()

    if story_select == 'Crew_movies_count':
        Crew_movies_count()

    if story_select == 'PieChart_noof_movies_by_Year':
        PieChart_noof_movies_by_Year()

    if story_select == 'Word_visualizations':
        Word_visualizations()

    if story_select == 'Genres_of_2000s_movies':
        Genres_of_2000s_movies()

    if story_select == 'Statistical_BoxOffice_by_Years':
        Statistical_BoxOffice_by_Years()

    if story_select == 'Duration_distribuion':
        Duration_distribuion()

    if story_select == 'Differentiation_scatters':
        Differentiation_scatters()

    if story_select == 'All Stories':
        Numberof_Movies_by_gcl()
        Year_vs_Movies()
        Max_BoxOffice_Movies_each_Year()
        Ratings_distribution()
        Maximum_Rated_Movies()
        Movies_based_datecount()
        Top_10_Movies_Titles()
        PieChart_noof_movies_by_Year()
        Word_visualizations()
        Genres_of_2000s_movies()
        Statistical_BoxOffice_by_Years()
        Duration_distribuion()
        Differentiation_scatters()
        Crew_movies_count()

def Visualizations():
    st.markdown("""### Visualizations of **Movies** data""")
    MDAV_options()

if story_select == 'Visualizations':
    Visualizations()
    
# hide_streamlit_style = """
# <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style>

# """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.sidebar.markdown('''
<small>Created by [M.Lakshmi Madhuri, T.Phaneendhar, A.Akhil, A.Sailaja, A.Vamsi Krishna, Y.Swapnith]
(https://github.com/Madhuri97/MoviesData_Analysis_Visualization_Team9).</small>''', unsafe_allow_html=True)