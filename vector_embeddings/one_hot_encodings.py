from sklearn.preprocessing import OneHotEncoder

categorical_data = [['Superhero'], ['Romance'], ['Action'], ['Romance']]

my_one_hot_encoder = OneHotEncoder(sparse=False)

my_one_hot_encoder.fit(categorical_data)

one_hot_encodings = my_one_hot_encoder.transform(categorical_data)

print(one_hot_encodings)