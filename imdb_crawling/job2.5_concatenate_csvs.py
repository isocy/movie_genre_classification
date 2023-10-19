import glob
import pandas as pd

csv_file_paths = glob.glob('./crawling_data/temp_*_movie_description.csv')

df_description_category = pd.DataFrame()
for csv_file_path in csv_file_paths:
    df_segment = pd.read_csv(csv_file_path)
    df_description_category = pd.concat([df_description_category, df_segment], axis='rows', ignore_index=True)
df_description_category.to_csv('./crawling_data/all_movie_descriptions.csv', index=False)
