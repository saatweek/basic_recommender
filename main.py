import kagglehub
import torch
import pandas as pd
import numpy as np
import os

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Reading all the datasets
credits = pd.read_csv(os.path.join(path, "credits.csv"))
keywords = pd.read_csv(os.path.join(path, "keywords.csv"))
links = pd.read_csv(os.path.join(path, "links.csv")) #only has movieID, imdbID, tmdbID. So, kinda useless
links_small = pd.read_csv(os.path.join(path, "links_small.csv"))
movies_metadata = pd.read_csv(os.path.join(path, "movies_metadata.csv"))
ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
ratings_small = pd.read_csv(os.path.join(path, "ratings_small.csv"))

# Some of the dataset has been weirdly formatted, let's investigate those and figure out a fix
# These datasets have problems : credits,