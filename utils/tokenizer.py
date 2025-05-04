import os
import pickle

class CharTokenizer:
    def __init__(self, text: str):
        # Egyedi karakterek (a vocab)
        chars = sorted(list(set(text))) #a set halmazt képez -> egyedi karakterek -> majd rendezzük
        self.vocab_size = len(chars) # ahány egyedi char olyan hosszú  a szótárnk

        # Karakter → token és token → karakter konverzió
        # Két alapvető szótár a tokenizációhoz
        self.stoi = { ch: i for i, ch in enumerate(chars) } # ez az egysoros Dict létrehozás -string (kulcs) to index: pl 'a' : 1
        self.itos = { i: ch for i, ch in enumerate(chars) } # itt meg az index a kulcs


    def encode(self, text: str):
        """Szöveg → token lista"""
        return [self.stoi[c] for c in text if c in self.stoi]
    # Ez a python nagyon tud halmozni...
    # Először a for végigmegy a texten visszaadva az egyes betűit, de csak azokat amik szerepelnek kulcsként a stoi szótárban ami egy bizt. ellenőrzés
    # Aztán visszaadja ezeknek az értékeit a self.stoi[c] -val meg a [..] egy új listába teszi

    def decode(self, tokens: list[int]):
        """Token lista → szöveg"""
        return ''.join([self.itos[i] for i in tokens])
    # ua mint az előző csak itt még összefűzés is van
