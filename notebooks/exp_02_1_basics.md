## 🧩 Az LLM tanítása: hogyan lesz a tokenből tudás?

Miután már megértettük, **mi az a tokenizáció**, és hogyan lehet egy szöveget **karakterekre bontani**, majd számokká alakítani (tokenek), érdemes megérteni, **hogyan tanul egy nyelvi modell** ezekből a tokenekből.

A tokenizálás önmagában még nem elég. Az csak azt oldja meg, hogy a nyelvet **átalakítjuk a gép számára feldolgozható formába**. De ettől még nem fogja *érteni* vagy *jósolni*, mi a következő szó egy mondatban, vagy hogy mit válaszoljon egy kérdésre.

Ehhez szükség van a **tanulásra**, vagyis arra, hogy a modell képes legyen **összefüggéseket felismerni a tokenek között**. Ezt a folyamatot mutatja be az alábbi kísérlet.

---

## 🎓 Miért ezzel kezdünk?

A nagy nyelvi modellek (LLM-ek) belsejében több millió (vagy milliárd) paraméter található, amelyek finoman szabályozzák, mit "gondol" a modell egy adott szövegdarabról. De hogy megértsük ezt a sokaságot, előbb meg kell tanulnunk **hogyan tanul egyetlen súly** — hiszen ez a legalapvetőbb építőeleme minden neurális modellnek.

Ez a kísérlet egyetlen bemenetet, egyetlen súlyt és egy célértéket használ, hogy megmutassa:

* Hogyan hoz létre egy modell egy kimenetet?
* Hogyan méri meg, hogy ez jó-e?
* Hogyan módosítja a saját viselkedését ennek alapján?

---

## 🔧 Fogalmak, amiket itt bevezetünk

### 1. **Súly (weight)** – ez az, amit a modell „megtanul”

Egy szám, ami meghatározza, hogyan hat a bemenet a kimenetre. A tanulás során ezt fogjuk finomhangolni.

### 2. **Előrefelé számolás (forward pass)** – kiszámoljuk a jelenlegi kimenetet

Például: ha a bemenet 3, és a súly 2, akkor `kimenet = 3 × 2 = 6`.

### 3. **Loss (hiba, veszteség)** – mennyire volt rossz a becslés?

A cél az, hogy a `kimenet` közel legyen a `célértékhez` (pl. 12). A loss azt méri, mekkora a különbség:

$$
\text{loss} = (kimenet - cél)^2
$$

### 4. **Visszaterjesztés (backpropagation)** – hogyan módosítsuk a súlyt?

A gradiens megmutatja: merre és milyen mértékben kell elmozdítani a súlyt, hogy kisebb legyen a hiba.

### 5. **Gradient Descent** – a súly tényleges frissítése

A gradiens irányában elmozdítjuk a súlyt, és újra számolunk. Ezt a folyamatot ismételjük, amíg a loss elég kicsi nem lesz.

---

## 📈 Hogyan kapcsolódik ez az LLM-ekhez?

Egy LLM belsejében:

* Minden tokenhez egy vektor (embedding) tartozik
* Ezeket a vektorokat **súlymátrixok** segítségével módosítjuk
* A súlyokat úgy tanítjuk, hogy **jósolni tudja a következő tokent**
* A tanulás alapja: pontosan **ugyanez a logika**, mint ebben az egyváltozós példában!

A különbség csak annyi, hogy ott:

* Több bemenet és súly van
* A kimenet vektor formájú (pl. valószínűségi eloszlás a szókészlet felett)
* Több rétegen (transformeren) keresztül történik a számítás

De **a matematikai alapelvek ugyanazok**: előrefelé számolás, loss, visszaterjesztés, súlyfrissítés.

---


