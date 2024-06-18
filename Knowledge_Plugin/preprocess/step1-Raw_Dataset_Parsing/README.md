We need to process the following data sets separately:

+ Amazon - Beauty (http://jmcauley.ucsd.edu/data/amazon/index.html)
+ MovieLens 1M Dataset (https://grouplens.org/datasets/movielens/1m/)
+ Online Retail (https://www.kaggle.com/carrie1/ecommerce-data)

# Download
Create the directory at Knowledge_Plugin/
```bash
mkdir data/raw_data
cd data/raw_data
# Amazon - Beauty
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz
wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz
# MovieLens 1M Dataset
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
# Online Retail
# Sign in Kaggle and follow the link in https://www.kaggle.com/carrie1/ecommerce-data to download the data.csv
```

# Process

Run each notebook according to the dataset.

+ data_preprocess_amazon.ipynb
+ data_preprocess_ml1m.ipynb
+ data_preprocess_onlineretail.ipynb

# Result

- sequential_data.txt: each line is `userid`, `itemids` seperated by `\t`.
- negative_samples.txt: sampled negative samples for each user.
- metadata.json: metadata of each item, including title, id, description, etc。