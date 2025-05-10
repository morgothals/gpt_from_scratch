## 🔤 Bevezetés a Tokenizációba: Mi az a Tokenizer, és miért van rá szükség?

Mielőtt egy nagy nyelvi modell (LLM) "megtanulhatná" az emberi nyelvet, először is értenie kell azt valamilyen *számokból álló formában*. A számítógépek nem tudnak közvetlenül a szavakkal vagy mondatokkal dolgozni — helyette számokká kell alakítani a szöveget. Ezt a folyamatot nevezzük **tokenizációnak**.

A tokenizáció során a szöveget **darabokra törjük** (ezek a darabok a *tokenek*), majd minden darabot hozzárendelünk egy számhoz. Ezek lesznek a bemenő adatok a nyelvi modell számára.

---

## 👶 Első lépés: Karakter-alapú tokenizálás

Ebben a kísérletben a szöveget **karakterekre bontjuk**. Ez a legegyszerűbb fajta tokenizálás:

* Minden egyes betű, szám, szóköz vagy írásjel egy külön token.
* Minden egyedi karakterhez egy számot (ID-t) rendelünk.

Ez még nem egy hatékony vagy „okos” módszer, de **nagyon jó tanulóeszköz**, mert pontosan megmutatja, hogy mit csinál egy tokenizer: szöveget → számokká alakít, majd vissza.

---

## 🧠 Mit csinál a kód?

1. **Beolvassa a szöveget** (pl. egy `.txt` fájlból).
2. **Létrehozza a tokenizer-t**: megkeresi az összes egyedi karaktert a szövegben, és mindegyikhez egyedi számot rendel.
3. **Encode**: egy példamondatot számokká alakít (`"Let them lead."` → `[5, 12, 30, …]`).
4. **Decode**: a tokenekből visszaállítja az eredeti szöveget.
5. **Mentés**:

   * A tokenizált szöveget bináris fájlba menti (`train.bin`, `val.bin`).
   * Magát a tokenizer-t is elmenti (`tokenizer.pkl`), hogy később újra felhasználható legyen.

---

## 🧱 Miért fontos ez?

Ez a kísérlet az LLM-ek *legelső építőkockáját* mutatja be:

* Ha nem tudjuk a szöveget számokká alakítani, a modell nem fog tudni sem tanulni, sem válaszolni.
* A karakter-tokenizálás megmutatja az alapelvet, amit később bonyolultabb (és hatékonyabb) módszerek váltanak fel, mint pl. a byte pair encoding (BPE), unigram, vagy a WordPiece.

---

## 📦 Mire jó a `tokenizer.pkl` és a `.bin` fájl?

* A `tokenizer.pkl` tartalmazza a teljes "szótárt" – azaz a karakter és szám közti párosításokat.
* A `train.bin` és `val.bin` tartalmazzák a szöveg tokenizált (számsorozattá alakított) változatát, amely készen áll arra, hogy egy modell tanuljon belőle.
