 **megépítjük a legegyszerűbb nyelvi modellt**, ami karakterekből próbálja megtanulni a következő karaktert **mindenféle Transformer vagy Attention nélkül**.

---

## 🎯 Cél: Első egyszerű nyelvi modell

Olyan modellt írunk, ami:

* bemenetként egy karakterláncot (pl. `"hel"`) kap,
* és megpróbálja megtippelni a következő karaktert (pl. `'l'` → `'o'`).

Ehhez csak két réteget használunk:

1. `Embedding` – a karaktereket vektorrá alakítja.
2. `Linear` – megjósolja a következő karakter logitjait.

---

Nagyon jó, hogy ennyire alaposan akarod megérteni — ez a legjobb módja annak, hogy ne csak használd a neurális hálókat, hanem **igazán értsd is, hogyan működnek belülről**. Kezdjük tényleg **nulláról**, és lépésről lépésre bontsuk ki:

---

# 🔥 1. Mire való a `torch` könyvtár?

A `torch`, vagyis **PyTorch**, egy **nyílt forráskódú gépi tanulási könyvtár**, amit a Meta (Facebook) fejleszt. Arra való, hogy:

| Feladat                      | Mire használható?                        |
| ---------------------------- | ---------------------------------------- |
| ⚙️ Mátrixműveletek           | Olyan, mint a `numpy`, de GPU-n is fut   |
| 🧠 Neurális hálók építése    | Rétegek, aktivációk, veszteségfüggvények |
| 🔁 Tanítás (backpropagation) | Automatikus gradiensszámítás             |
| 🧪 Modell tanítása           | Optimizer, training loop, loss számítás  |
| 📦 Modell mentés/betöltés    | Tréning után elmented a súlyokat         |

Gyakorlatilag: **ez az az eszköz, amivel saját neurális hálókat tudsz írni és edzeni.**

---

# 🧱 2. Mi van a `torch.nn` modulban?

A `torch.nn` az **ideghálózat-építő modul** (nn = neural network). Tartalmaz mindent, amiből hálót tudsz építeni:

| Tartalom            | Példa                                               |
| ------------------- | --------------------------------------------------- |
| Rétegek             | `nn.Linear`, `nn.Embedding`, `nn.Conv2d`, `nn.LSTM` |
| Aktivációk          | `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`                  |
| Normálások          | `nn.LayerNorm`, `nn.BatchNorm1d`                    |
| Veszteségfüggvények | `nn.CrossEntropyLoss`, `nn.MSELoss`                 |
| Komplex egységek    | `nn.Transformer`, `nn.Sequential`                   |

Ezeket használod úgy, mint LEGO-kockákat: **összerakod őket egy nagy modellé**.

---

# 🧬 3. Mi az a `nn.Module`, és miért öröklünk belőle?

A `nn.Module` az **összes hálózat alaposztálya**. Ez tartalmazza:

| Funkció             | Miért fontos?                                      |
| ------------------- | -------------------------------------------------- |
| `__init__()`        | Ide rakod a rétegeidet                             |
| `forward()`         | Itt határozod meg, hogyan halad át az adat a hálón |
| `parameters()`      | Automatikusan tudja, mik a tanítható súlyok        |
| `to(device)`        | Áthelyezés CPU ↔ GPU között                        |
| `eval()`, `train()` | Váltás edző / kiértékelő mód között                |
| `state_dict()`      | Mentéshez, betöltéshez szükséges belső állapot     |

### 🔍 Mi történne, ha nem örökölnénk?

Akkor:

* nem lenne egységes struktúrája a modellednek,
* nem tudnád `optimizer.step()`-pel frissíteni a súlyokat,
* nem lehetne `model(x)` módon meghívni a hálót,
* nem tudnád lementeni `.pt` fájlba.

Tehát a `nn.Module` az **a "keret", ami köré a tanulható, menthető és használható neurális háló épül**.

---

## 🛠️ Mi történik valójában?

Amikor ezt írod:

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 20)

    def forward(self, x):
        return self.layer(x)
```

akkor a PyTorch:

* automatikusan elmenti a `self.layer` paramétereit,
* ismeri a `.forward()` utasításaidat,
* tudja majd, hogy ezt egy teljes modellként kell kezelni,
* és integrálja az **optimizálás**, **mentés**, **inferencia** rendszerébe.

---

# ✅ Összefoglalás

| Fogalom                  | Mire való?                                                                                |
| ------------------------ | ----------------------------------------------------------------------------------------- |
| `torch`                  | Gépi tanulási alap – számítás, tensor, GPU                                                |
| `torch.nn`               | Rétegek, aktivációk, veszteségek, modellek                                                |
| `nn.Module`              | A modell „váza” – összefogja a rétegeket és tud tanulni                                   |
| Miért nem írunk sajátot? | Mert a `nn.Module` már tud mindent, amire szükség van (súlykezelés, tanítás, mentés stb.) |

---

Szeretnéd, hogy most mutassak egy konkrét példát arra, hogyan működik egy teljesen minimális `nn.Module` osztály, és hogyan történik benne először egy „előrehaladás” (forward pass)?




## 💻 Kód: `bigram_model.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Ez lesz az embedding és egyben a kimenet is
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T) – batch, time
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Átalakítjuk a bemeneteket lapos tömbbé (batch*T), hogy loss-t számoljunk
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # csak az utolsó tokenre
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
```

---

## 📦 Mi történik itt?

| Rész                                   | Mit csinál?                                                   |
| -------------------------------------- | ------------------------------------------------------------- |
| `nn.Embedding(vocab_size, vocab_size)` | Minden tokennek egy saját „jósló” vektora van                 |
| `forward()`                            | Visszaadja a logitokat, és ha van `targets`, loss-t is számol |
| `generate()`                           | Karaktereket generál a modellből                              |

Ez még **nem „tanul”, csak definiálja a hálót**.

---

## 📁 Lépés 2: egyszerű tanító szkript

Következő lépésben írunk hozzá egy `train/train_bigram.py` fájlt, ami:

* betölti az `input.txt`-et,
* tokenizál,
* minibatch-eken edzi a modellt,
* és néha karaktereket generál.

---

👉 Készen állsz erre a következő lépésre? Vagy előbb szeretnéd, hogy ezt a `BigramModel` osztályt részletesen, soronként elmagyarázzam?
