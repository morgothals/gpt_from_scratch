import os
import pickle
import sys


# A projekt gyökérkönyvtárának abszolút elérési útja
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from utils.tokenizer import CharTokenizer

# Szöveg beolvasása
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenizer létrehozása
tokenizer = CharTokenizer(text)

# Tokenizálás
data = tokenizer.encode(text)
print(f"Összes token: {len(data)}")

# Train / val split (90% - 10%)
split = int(len(data) * 0.9)
train_data = data[:split]
val_data = data[split:]

# Bináris fájlokba mentés
os.makedirs("data/out", exist_ok=True)

with open("data/out/train.bin", "wb") as f:
    f.write(bytearray(train_data))

with open("data/out/val.bin", "wb") as f:
    f.write(bytearray(val_data))

# Tokenizer elmentése (pickle)
with open("data/out/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Előkészítés kész: train.bin, val.bin és tokenizer.pkl elmentve.")
