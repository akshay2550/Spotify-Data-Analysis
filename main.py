import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_tracks = pd.read_csv('tracks.csv')

sorted_df = df_tracks.sort_values(by='popularity',ascending=True).head(10)

most_popular_songs = df_tracks[df_tracks['popularity'] > 90].sort_values('popularity',ascending=False)

df_tracks.set_index('release_date',inplace=True)
df_tracks.index = pd.to_datetime(df_tracks.index)

df_tracks['duration'] = df_tracks['duration_ms'].apply(lambda x: round(x/1000))
df_tracks.drop('duration_ms',inplace=True,axis=1)

corr_df = df_tracks.drop(['key','mode','explicit'],axis=1).corr(method='pearson')
plt.figure(figsize=(14,6))
heatmap = sns.heatmap(corr_df,annot=True, vmin=-1,vmax=1,cmap='inferno',linewidths=1,linecolor='Black')
heatmap.set_title('Correlation HeatMap Between Variable')
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)

sample_df = df_tracks.sample(int(0.004*len(df_tracks)))

plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y='loudness',x='energy',color='c').set_title('Loudness vs Energy Correlation')

plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y='popularity',x='acousticness',color='b').set_title('Popularity vs Acousticness Correlation')

df_tracks['dates'] = df_tracks.index.get_level_values('release_date')
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year

sns.displot(years,discrete=True,aspect=2,height=5,kind='hist').set(title="Number of Songs per year")

total_dr = df_tracks.duration

fig,axes = plt.subplots(figsize = (18,7))
fig = sns.barplot(x=years,y=total_dr,ax=axes,errwidth=False).set(title='Year vs Duration')
plt.xticks(rotation=90)

plt.show()