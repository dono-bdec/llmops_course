from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Create a dataframe of our data
movies_df = pd.DataFrame({
    "name": ["Avengers", "John Wick", "The Time Traveller's Wife", "Bridget Jones' Diary"],
    "genre": ["Superhero", "Action", "Romance", "Romance"],
    "director": ["Joss Whedon", "Chad Stahelski", "Robert Schwentke", "Sharon Maguire"]
    })

# print(df)

columns_to_be_used_for_encoding = ["genre", "director"]
newly_encoded_columns = []

# Get the columns that will be needed for encoding
for col in columns_to_be_used_for_encoding:
    newly_encoded_columns += [f"is_{category}" for category in movies_df[col].unique().tolist()]

# Check out the new column names after encoding
print(newly_encoded_columns)


# Use OneHotEncoder to create the encodings
my_one_hot_encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
encoded_cols = my_one_hot_encoder.fit_transform(movies_df[columns_to_be_used_for_encoding])


# Conver the sparse matrix into a dataframe
final_df_encoded = pd.DataFrame(encoded_cols, columns = newly_encoded_columns)
final_df_one_hot_encoded = movies_df.join(final_df_encoded)



print(final_df_one_hot_encoded)