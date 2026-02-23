import os
import math
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
urllib.request.urlretrieve(url, "ml-1m.zip")

with zipfile.ZipFile("ml-1m.zip", "r") as zip_ref:
    zip_ref.extractall()

users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    engine="python",
    encoding="ISO-8859-1",
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "timestamp"],
    engine="python",
    encoding="ISO-8859-1",
)

movies = pd.read_csv(
    "ml-1m/movies.dat",
    sep="::",
    names=["movie_id", "title", "genres"],
    engine="python",
    encoding="ISO-8859-1",
)


users["user_id"] = users["user_id"].astype(str)
movies["movie_id"] = movies["movie_id"].astype(str)
ratings["movie_id"] = ratings["movie_id"].astype(str)
ratings["user_id"] = ratings["user_id"].astype(str)
ratings["rating"] = ratings["rating"].astype(float)



GENRES = [
    "Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

for g in GENRES:
    movies[g] = movies["genres"].apply(lambda x: int(g in x.split("|")))
movie_genre_matrix = movies[GENRES].values



SEQUENCE_LENGTH = 4
STEP_SIZE = 2

def create_sequences(values, window, step):
    out = []
    i = 0
    while True:
        chunk = values[i:i+window]
        if len(chunk) < window:
            if len(values) >= window:
                out.append(values[-window:])
            break
        out.append(chunk)
        i += step
    return out



ratings = ratings.sort_values("timestamp")
grouped = ratings.groupby("user_id")

data = []
for uid, g in grouped:
    movies_seq = create_sequences(g.movie_id.tolist(), SEQUENCE_LENGTH, STEP_SIZE)
    ratings_seq = create_sequences(g.rating.tolist(), SEQUENCE_LENGTH, STEP_SIZE)
    for m, r in zip(movies_seq, ratings_seq):
        data.append((uid, m[:-1], r[:-1], m[-1], r[-1]))

df = pd.DataFrame(
    data,
    columns=["user_id","seq_movies","seq_ratings","target_movie","target_rating"]
)

user2idx = {u:i for i,u in enumerate(users.user_id.unique())}
movie2idx = {m:i for i,m in enumerate(movies.movie_id.unique())}

df["user_id"] = df["user_id"].map(user2idx)
df["target_movie"] = df["target_movie"].map(movie2idx)
df["seq_movies"] = df["seq_movies"].apply(lambda x: [movie2idx[m] for m in x])


class MovieDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "user": torch.tensor(row.user_id),
            "seq_movies": torch.tensor(row.seq_movies),
            "seq_ratings": torch.tensor(row.seq_ratings, dtype=torch.float),
            "target_movie": torch.tensor(row.target_movie),
            "target": torch.tensor(row.target_rating, dtype=torch.float)
        }


train_df = df.sample(frac=0.85)
test_df = df.drop(train_df.index)

train_loader = DataLoader(MovieDataset(train_df), batch_size=256, shuffle=True)
test_loader = DataLoader(MovieDataset(test_df), batch_size=256)



class BST(nn.Module):
    def __init__(self, num_movies, embed_dim=64, num_heads=4):
        super().__init__()

        self.movie_emb = nn.Embedding(num_movies, embed_dim)
        self.pos_emb = nn.Embedding(SEQUENCE_LENGTH-1, embed_dim)

        self.genre_emb = nn.Embedding.from_pretrained(
            torch.tensor(movie_genre_matrix, dtype=torch.float),
            freeze=True
        )
        self.genre_fc = nn.Linear(embed_dim + len(GENRES), embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * SEQUENCE_LENGTH, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def encode_movie(self, ids):
        emb = self.movie_emb(ids)
        genre = self.genre_emb(ids)
        return self.genre_fc(torch.cat([emb, genre], dim=-1))

    def forward(self, batch):
        seq = batch["seq_movies"]
        ratings = batch["seq_ratings"].unsqueeze(-1)

        x = self.encode_movie(seq)
        pos = self.pos_emb(torch.arange(seq.size(1), device=seq.device))
        x = (x + pos) * ratings

        target = self.encode_movie(batch["target_movie"].unsqueeze(1))
        x = torch.cat([x, target], dim=1)

        x = self.transformer(x)
        x = x.flatten(1)
        return self.fc(x).squeeze()



device = "cuda" if torch.cuda.is_available() else "cpu"
model = BST(len(movie2idx)).to(device)

optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()


for epoch in range(5):
    model.train()
    for batch in tqdm(train_loader):
        batch = {k:v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch["target"])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Loss:", loss.item())

for epoch in range(5):
    model.train()
    for batch in tqdm(train_loader):
        batch = {k:v.to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch["target"])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch +1 } Loss: {loss.item()}")

