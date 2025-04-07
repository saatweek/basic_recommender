print("Importing all libraries...")
import kagglehub
import torch
import pandas as pd
import numpy as np
import os
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn import model_selection, preprocessing, metrics
import torch.nn as nn
import matplotlib.pyplot as plt
import math

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
print("Imported!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for processing")

# Download dataset
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Load data
movies_metadata = pd.read_csv(os.path.join(path, "movies_metadata.csv"))
ratings = pd.read_csv(os.path.join(path, "ratings.csv"))

# Pre-specified genre list
GENRES = ["drama", "comedy", "action", "romance", "documentary",
          "thriller", "adventure", "fantasy", "crime", "horror"]


# Encoding functions
def encode_genres(s: str) -> np.ndarray:
    genre_list = []
    try:
        genres = ast.literal_eval(s)
        for item in genres:
            genre_list.append(item['name'].lower())
    except:
        pass

    encoding = [0] * len(GENRES)
    for genre in genre_list:
        if genre in GENRES:
            encoding[GENRES.index(genre)] = 1
    return np.array(encoding)


def encode_lang(lang: str) -> np.ndarray:
    lang = lang.lower()
    if lang == "en":
        return np.array([1, 0, 0, 0])
    elif lang == "fr":
        return np.array([0, 1, 0, 0])
    elif lang == "es":
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])  # Other category


def encode_vote_count(num: int) -> np.ndarray:
    bins = [0] * 10
    try:
        idx = min(math.floor(num / 100), 9)
        bins[idx] = 1
    except:
        bins[-1] = 1
    return np.array(bins)


def encode_vote_avg(avg: float) -> np.ndarray:
    if avg < 2:
        return np.array([1, 0, 0])
    elif avg < 4:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


# Preprocessing
print("Processing data...")
movies_metadata['id'] = pd.to_numeric(movies_metadata['id'], errors='coerce')

# Handle missing values
print("handling missing values...")
movies_metadata = movies_metadata.dropna(subset=['id'])
movies_metadata['genres'] = movies_metadata['genres'].fillna('[]')
movies_metadata['original_language'] = movies_metadata['original_language'].fillna('en')
movies_metadata['vote_count'] = movies_metadata['vote_count'].fillna(0).astype(int)
movies_metadata['vote_average'] = movies_metadata['vote_average'].fillna(0)

# Filter top users
print("Filtering top users...")
user_counts = ratings['userId'].value_counts()
top_users = user_counts[user_counts > 100].index  # More robust filtering
merged = ratings[ratings['userId'].isin(top_users)].merge(
    movies_metadata, left_on='movieId', right_on='id', how='inner')

# Apply encodings
print("Encoding features...")
merged['genres_enc'] = merged['genres'].apply(encode_genres)
merged['lang_enc'] = merged['original_language'].apply(encode_lang)
merged['vote_count_enc'] = merged['vote_count'].apply(encode_vote_count)
merged['vote_avg_enc'] = merged['vote_average'].apply(encode_vote_avg)
merged['target'] = (merged['rating'] >= 3).astype(int)  # Binary classification target

# Label encoding
user_encoder = preprocessing.LabelEncoder()
movie_encoder = preprocessing.LabelEncoder()
merged['user_id'] = user_encoder.fit_transform(merged['userId'])
merged['movie_id'] = movie_encoder.fit_transform(merged['movieId'])


# Dataset class
class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movie_id'].values, dtype=torch.long)
        self.genres = torch.tensor(np.stack(df['genres_enc'].values), dtype=torch.float32)
        self.lang = torch.tensor(np.stack(df['lang_enc'].values), dtype=torch.float32)
        self.vote_count = torch.tensor(np.stack(df['vote_count_enc'].values), dtype=torch.float32)
        self.vote_avg = torch.tensor(np.stack(df['vote_avg_enc'].values), dtype=torch.float32)
        self.targets = torch.tensor(df['target'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return {
            'user': self.users[idx],
            'movie': self.movies[idx],
            'genres': self.genres[idx],
            'lang': self.lang[idx],
            'vote_count': self.vote_count[idx],
            'vote_avg': self.vote_avg[idx],
            'target': self.targets[idx]
        }


# Model architecture
print("Initialized model..")
class RecSysModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, 32)
        self.movie_emb = nn.Embedding(num_movies, 32)

        # Feature dimensions: genres(10) + lang(4) + vote_count(10) + vote_avg(3) = 27
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 27, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user, movie, genres, lang, vote_count, vote_avg):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        features = torch.cat([u, m, genres, lang, vote_count, vote_avg], dim=1)
        return self.fc(features)


# Train/Test split
train_df, val_df = model_selection.train_test_split(
    merged, test_size=0.2, stratify=merged['target'], random_state=42)

train_ds = MovieDataset(train_df)
val_ds = MovieDataset(val_df)

print("Loading data to model..")
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)

# Initialize model
model = RecSysModel(
    num_users=len(user_encoder.classes_),
    num_movies=len(movie_encoder.classes_)
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
epochs = 10
losses = []
print("Training started...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
        output = model(**inputs)
        loss = criterion(output.squeeze(), batch['target'].to(device))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot training loss
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.show()

# Evaluation
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for batch in val_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'target'}
        outputs = model(**inputs).squeeze().sigmoid()
        val_preds.extend(outputs.cpu().numpy())
        val_targets.extend(batch['target'].cpu().numpy())

print(f"Validation ROC-AUC: {metrics.roc_auc_score(val_targets, val_preds):.4f}")
print(f"Validation Accuracy: {metrics.accuracy_score(val_targets, np.round(val_preds)):.4f}")

# Example prediction
test_sample = next(iter(val_loader))
with torch.no_grad():
    inputs = {k: v.to(device) for k, v in test_sample.items() if k != 'target'}
    outputs = model(**inputs).sigmoid()
    print("\nSample predictions:")
    print("Predicted:", outputs[:5].cpu().numpy().flatten())
    print("Actual:   ", test_sample['target'][:5].numpy())