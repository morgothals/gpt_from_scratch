## 🧠 Mi az a PyTorch, és mit nyújt nekünk?

A PyTorch egy gépi tanulási könyvtár, amely:

* **Többdimenziós tömböket kezel (tensort)**
* **Automatikusan számolja a gradienset** (backpropagation)
* **Segít a modell súlyainak frissítésében**

Ahelyett, hogy minden számítást és deriváltat kézzel írnánk meg, a PyTorch automatikusan:

* Épít egy **számítási gráfot** a műveleteinkről
* Ebben a gráfban **nyomon követi**, mit hogyan számoltunk
* Ez alapján **visszaterjeszti a hibát**, és kiszámolja, hogyan változzanak a tanulható paraméterek

---

## 🔍 A kísérletben szereplő PyTorch-fogalmak röviden

| Fogalom                | Mit csinál?                                                                |
| ---------------------- | -------------------------------------------------------------------------- |
| `torch.tensor(...)`    | Egy szám vagy mátrix PyTorch-verziója – az alapadat.                       |
| `requires_grad=True`   | Jelzi: ez a tensor tanulható, számoljunk rá gradienset.                    |
| `.backward()`          | Elindítja a visszaterjesztést: kiszámolja, hogyan kell módosítani a súlyt. |
| `.grad`                | A számított gradiens (irány és mérték) – ez alapján frissítjük a súlyt.    |
| `with torch.no_grad()` | Kikapcsoljuk a gráfépítést, hogy kézzel módosíthassuk a súlyt.             |
| `.data`                | A tensor „nyers értéke” – ezt módosítjuk közvetlenül frissítéskor.         |

---

## 📈 Példa: egyetlen súly tanítása – kézzel vs PyTorch-szal

Korábban, ha egy `W` súlyt tanítani akartunk, ezt kellett tennünk:

1. Kiszámolni `y = W * x`
2. Loss: `(y - target)^2`
3. Deriválni kézzel: `dL/dW = 2 * (y - target) * x`
4. Frissíteni: `W = W - lr * gradiens`

**A PyTorch ezt mind elvégzi helyettünk.** Csak annyit mondunk:

```python
loss.backward()
W.data -= lr * W.grad
```

És kész.

---

## 📌 Miért nem jó, ha *mindent* kézzel csinálunk?

* ❗ **Bonyolultabb modelleknél** a kézi deriválás hibalehetőséget rejt.
* 📉 Lassú fejlesztés: minden új változatnál új kézi számítást kéne írni.
* 🔁 Nehéz debugolni vagy újrakísérletezni.
* ⚙️ A PyTorch automatikusan tud GPU-n futni, skálázható nagy modellekre is.

Ezért már a legelső tanulási példáknál érdemes **PyTorch-ot használni**: nem lustaságból, hanem hogy:

* **jobban fókuszálhassunk a modellre és a viselkedésére**, és ne a matematikai deriválásra,
* **automatizáljuk** azt, amit a gép úgyis jobban tud csinálni.

---

## 🚀 Hogyan fog ez tovább bővülni?

Később a `torch.tensor` és `requires_grad=True` lesz a belépőnk a következő szintekre:

* **több bemenet**, több súly (pl. `W = torch.randn(3, 1)`)
* **nemlineáris aktivációk** (pl. `tanh`, `relu`)
* **hálózatok** (`nn.Linear`, `nn.Sequential`, saját modellek)
* **komplex tanítási ciklusok** (batch, epoch, optimizer)

Minden ezekre az alapokra épül.

---

## 🧠 Összefoglalás

| Fogalom                | Lényeg                                           |
| ---------------------- | ------------------------------------------------ |
| PyTorch tensor         | Többdimenziós adatstruktúra tanuláshoz           |
| Autograd               | Automatikusan számol gradienset                  |
| `requires_grad=True`   | Figyeld ezt a tensort, hogy taníthassuk          |
| `loss.backward()`      | Gráf mentén visszaterjeszti a hibát              |
| `W.grad`               | A gradiens: hogyan változtassuk a súlyt          |
| `with torch.no_grad()` | Kikapcsoljuk a gráfot, amikor frissítjük a súlyt |





## 🧬 PyTorch különlegességei – amiért *több*, mint egy „mátrixkönyvtár”

A PyTorch a gépi tanulásra tervezett egyik legfontosabb eszköz. Első ránézésre hasonlít a `numpy`-ra vagy más tömbkezelő könyvtárakra, de **három nagyon fontos dologban különleges**:

---

### 📌 1. **Tensor ≠ csak mátrix**

A **tensor** a PyTorch alapegysége. Ez egy *tetszőleges dimenziószámú numerikus tömb*. Lehet:

* skalár (0D): pl. `torch.tensor(2.0)`
* vektor (1D): pl. `[1.0, 2.0, 3.0]`
* mátrix (2D): pl. `[[1, 2], [3, 4]]`
* vagy ennél többdimenziós, pl. `[batch, channels, height, width]` (4D)

> **Tensors = általánosított mátrixok**. A mátrix csupán egy speciális eset: 2D tensor.

---

### 📌 2. **Egy elemű tensor = tanítható skalár**

Ha azt szeretnénk, hogy egyetlen szám (pl. egy súly) **tanítható** legyen, akkor **tensor-ként kell reprezentálnunk**, nem sima float-ként. Ezért írunk ilyet:

```python
W = torch.tensor(2.0, requires_grad=True)
```

Ez egy *egy elemű tensor* (`shape = torch.Size([])`), de mivel `requires_grad=True`, a PyTorch **gráfot épít köré**, és **visszaterjeszthetővé** teszi.

---

### 🧠 3. **Automatikus számítási gráf (autograd)**

A PyTorch automatikusan nyomon követi a számításokat a `requires_grad=True` tensorokon keresztül. Minden művelet:

* nemcsak kiszámítja az eredményt,
* hanem **nyilvántartja a műveletet is**, és létrehoz egy **gráf csomópontot** (pl. `MulBackward0`).

Ez az úgynevezett **dinamikus számítási gráf**:

* A gráf **futás közben épül** (nem előre definiált).
* Ez rugalmas és könnyen debuggolható, szemben pl. TensorFlow 1 statikus gráfjával.

---

### 🔁 4. Mi történik `loss.backward()` során?

1. A `loss` tensor egy `grad_fn` objektumot tartalmaz (pl. `PowBackward0`).
2. Ez a `grad_fn` tudja, hogyan számolja ki a deriváltat az előző lépések szerint.
3. A `loss.backward()` hívás:

   * Elindul ezen a gráfon visszafelé,
   * Minden `requires_grad=True` tensorra kiszámolja: ∂loss/∂paraméter
   * A gradiens az adott tensor `.grad` mezőjébe kerül.

---

### 🔒 5. Miért kell kikapcsolni a gráfot súlyfrissítéskor?

A következő kód:

```python
W = W - learning_rate * W.grad
```

**új számítást hajt végre**, így a PyTorch **gráfot építene erről is**. Ez hibás lenne, mert:

* Felhalmozódó gráfrészeket eredményez,
* Lassítja a futást, sőt: később **hibát is dobhat**.

Ezért használjuk:

```python
with torch.no_grad():
    W.data = W.data - learning_rate * W.grad
```

Ez azt mondja: *"Most csak frissítek. Ne építs új gráfot."*

---

### 🧹 6. Gradiens kinullázása: `.grad.zero_()`

A `.backward()` minden hívása **hozzáadja** a gradiens értékét a tensor `.grad` mezőjéhez. Ezért **minden frissítés után nullázni kell**, különben a gradiens felhalmozódik:

```python
W.grad.zero_()
```

---

### 📐 7. Hogyan épül fel a gráf?

Példa:

```python
W = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = W * x          # MulBackward0
loss = (y - 12)**2 # PowBackward0
```

A PyTorch ilyenkor ezt jegyzi meg:

```
W --Mul--> y --Sub--> (y-12) --Pow--> loss
```

Minden művelethez tartozik egy **gradiens-számítási szabály**, amit visszafele alkalmaz:

* `∂loss/∂y = 2*(y - 12)`
* `∂loss/∂W = ∂loss/∂y * ∂y/∂W = 2*(y - 12) * x`

---




