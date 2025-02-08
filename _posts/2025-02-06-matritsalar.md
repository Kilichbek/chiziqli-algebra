---
title: Modul 2. Matritsa Asoslari
author: Qilichbek Haydarov
date: 2025-01-13
category: Jekyll
layout: post
---

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/0AWdezwHRY0?si=VhIDHwGiUGmAqOSn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</center>

> ##### Matritsa
>
> $m$ ta qator va $n$ ta ustundan tashkil topgan sonlarning ro'yxati.
{: .block-tip }

Matritsalar odatda kvadrat yoki oddiy qavslar bilan o'ralgan jadvallar bo'lib ular ushbu ko'rinishda yoziladi:


$$ \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{bmatrix}$$ yoki $$ \mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{pmatrix}$$

A - matritsa, $a_{ij}$ - matritsa elementlari, $i$ - qator, $j$ esa ustun raqamini ko'rsatadi.

```python
import numpy as np

# Matritsa yaratish
matritsa = np.array(
    [
        [-1.1, 0.0, 3.6],
        [1.1, 0.0, 3.6],
        [1.1, 0.0, 3.6]
    ]
)
print("Matritsa: ", matritsa)
print("Matritsa o'lchami: ", matritsa.shape)
print("1-qator, 3-chi ustunda joylashgan element: ", matrits[0, 2])
print("Matritsaning 2-qatori: ", matritsa[1, :])
print("Matritsaning 2-ustuni: ", matritsa[:, 1])
```

2.1 Matritsa turlari
-------------
Matritsalar ko'p turli bo'lishi mumkin. Ammo, ularning asosiy turlari quyidagilardir:

### 2.1.1 Kvadrat matritsa
$ n \times n $ o'lchamli matritsa. Bu matritsa qator va ustunlar soni bir xil bo'lgan matritsa. Misol uchun, 

$$ \mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

### 2.1.2 To'g'ri to'rtburchak matritsa
$ m \times n $ o'lchamli matritsa. Bu matritsa qator va ustunlar soni bir xil bo'lmagan matritsa. Misol uchun, 

$$ \mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$

### 2.1.3 Nol matritsa
Barcha elementlari nol bo'lgan matritsa. Misol uchun,

$$ \mathbf{A} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}$$

### 2.1.4 Diagonal matritsa
Kvadrat matritsa bo'lib, diagonal elementlari nol bo'lmagan matritsa. Misol uchun,

$$ \mathbf{A} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix}$$

### 2.1.5 Birlik matritsa

Diagonal elementlari bir bo'lgan diagonal matritsa. Misol uchun,

$$ \mathbf{I} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### 2.1.6 Simmetrik matritsa
Diagonal bo'yicha bu matritsaning elementlari simmetrik bo'lgan matritsa. Ya'ni, $a_{ij} = a_{ji}$ shartni qanoatlantiradigan matritsa. Misol uchun,

$$ \mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix}$$


### 2.1.7 Uchburchak matritsa
Uchburchak matritsalar ikki turdagi bo'lishi mumkin:
- Yuqoridan uchburchak matritsa
- Quyidan uchburchak matritsa

Yuqoridan uchburchak matritsa bu diagonal elentlaridan pastki elementlari nol bo'lgan matritsa. Misol uchun,

$$ \mathbf{U} = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}$$

Quyidan uchburchak matritsa bu diagonal elentlaridan yuqori elementlari nol bo'lgan matritsa. Misol uchun,

$$ \mathbf{L} = \begin{bmatrix} 1 & 0 & 0 \\ 2 & 4 & 0 \\ 3 & 5 & 6 \end{bmatrix}$$



2.2 Matritsa ustida amallar
-------------

Chiziqli algebrada matritsalarnig ustida:
- qo'shish
- matritsani songa ko'paytirish
- matritsani vektorga ko'paytirish
- matritsani matritsaga ko'paytirish

kabi amallar bajariladi.

### 2.2.1 Matritsalarni Qo'shish

Ikkita $m \times n$ o'lchamli matritsani (masalan, $\mathbf{A}$ va $\mathbf{B}$ ) bir biriga qo'shish uchun ularning mos elementlarini qo'shish kerak.


$$ \mathbf{A} + \mathbf{B} = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} & \ldots & a_{1n} + b_{1n} \\ a_{21} + b_{21} & a_{22} + b_{22} & \ldots & a_{2n} + b_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} + b_{m1} & a_{m2} + b_{m2} & \ldots & a_{mn} + b_{mn} \end{bmatrix}$$


Misol uchun, ikkita matritsaning qo'shish amali quyidagicha: agar $$ \mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{va} \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

$$ \mathbf{A} + \mathbf{B} = \begin{bmatrix} 1+5 & 2+6 \\ 3+7 & 4+8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$$


> ##### Diqqat!
>
>Faqat bir hil o'lchamli matritsalarni qo'shish mumkin
{: .block-warning }


```python
import numpy as np

# Matritsa yaratish
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
# Matritsalarni qo'shish
C = A + B
print(C)
# yoki
C = np.add(A, B)
print(C)
```

### 2.2.2 Matritsani songa ko'paytirish

Matritsani ma'lum bir songa ($\lambda \in \mathbb{R}$) ko'paytirish uchun, matritsani har bir elementini ushbu songa ko'paytiramiz.

$$ \lambda \mathbf{A} = \begin{bmatrix} \lambda a_{11} & \lambda a_{12} & \ldots & \lambda a_{1n} \\ \lambda a_{21} & \lambda a_{22} & \ldots & \lambda a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ \lambda a_{m1} & \lambda a_{m2} & \ldots & \lambda a_{mn} \end{bmatrix}$$



Misol uchun, agar $\lambda = 3$ bo'lsa va $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$ bo'lsa,

$$ \lambda \mathbf{A} = 3 \cdot \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 9 & 12 \end{bmatrix}$$


```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
l = 3
# Matritsani songa ko'paytirish
C = l * A
print(C)
```

### 2.2.3 Matritsani vektorga ko'paytirish
Matritsani vektorga ko'paytirish uchun, matritsaning har bir qatorini vektorga skalyar ko'paytiramiz, va natijada yangi vektor hosil qilamiz.

$$ \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn} \end{bmatrix} \quad \text{va} \quad \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$


$$ \mathbf{A} \mathbf{v} = \begin{bmatrix} a_{11} v_1 + a_{12} v_2 + \ldots + a_{1n} v_n \\ a_{21} v_1 + a_{22} v_2 + \ldots + a_{2n} v_n \\ \vdots \\ a_{m1} v_1 + a_{m2} v_2 + \ldots + a_{mn} v_n \end{bmatrix} = \begin{bmatrix} \sum_{j=1}^{n} a_{1j} v_j \\ \sum_{j=1}^{n} a_{2j} v_j \\ \vdots \\ \sum_{j=1}^{n} a_{mj} v_j \end{bmatrix} $$

Misol, agar $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{va} \quad \mathbf{v} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$ bo'lsa,

$$ \mathbf{A} \mathbf{v} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 6 \\ 3 \cdot 5 + 4 \cdot 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$


```python
import numpy as np

v = np.array([1, 2, 3])
A = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])

B = np.dot(A, v)
print(B)

# yoki
B = A @ v
print(B)
```

### 2.2.4 Matritsani matritsaga ko'paytirish

Matritsani matritsaga ko'paytirish uchun, matritsaning har bir qatorini ikkinchi matritsaning ustuniga ko'paytiramiz, va natijada yangi matritsa hosil qilamiz.

$$ \mathbf{A} = \begin{bmatrix} a_{11} & \ldots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{m1} & \ldots & a_{mn} \end{bmatrix} \quad \text{va} \quad \mathbf{B} = \begin{bmatrix} b_{11} & \ldots & b_{1p} \\ \vdots & \ddots & \vdots \\ b_{n1} & \ldots & b_{np} \end{bmatrix}$$

$$ \mathbf{A} \mathbf{B} = \begin{bmatrix} a_{11} b_{11} + \ldots + a_{1n} b_{n1} & \ldots & a_{11} b_{1p} + \ldots + a_{1n} b_{np} \\ \vdots & \ddots & \vdots \\ a_{m1} b_{11} + \ldots + a_{mn} b_{n1} & \ldots & a_{m1} b_{1p} + \ldots + a_{mn} b_{np} \end{bmatrix}$$

Misol, agar $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{va} \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$ bo'lsa,

$$ \mathbf{A} \mathbf{B} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$


> ##### Diqqat!
>
>Matritsalar ko'paytirish amaliyotlari uchun matritsalar o'lchamlari mos kelishi kerak. Agar $\mathbf{A}$ matritsasi $m \times n$ o'lchamli bo'lsa, $\mathbf{B}$ matritsasi $n \times p$ o'lchamli bo'lishi kerak. Ya'ni, $\mathbf{A}$ matritsasining ustunlar soni $\mathbf{B}$ matritsasining qatorlar soniga teng bo'lishi kerak.
{: .block-warning }


```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8], [9, 10]])

C = np.dot(A, B)
print(C)

# yoki
C = A @ B
print(C)
```


### 2.2.5 Hadamard ko'paytirish

Matritsalarni Hadamard ko'paytirish uchun, ularning shunchaki mos elementlarini ko'paytiramiz.

$$\mathbf{A} \odot \mathbf{B} = \begin{bmatrix} a_{11} \cdot b_{11} & a_{12} \cdot b_{12} & \ldots & a_{1n} \cdot b_{1n} \\ a_{21} \cdot b_{21} & a_{22} \cdot b_{22} & \ldots & a_{2n} \cdot b_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} \cdot b_{m1} & a_{m2} \cdot b_{m2} & \ldots & a_{mn} \cdot b_{mn} \end{bmatrix}$$

Misol uchun, agar $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{va} \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$ bo'lsa,

$$\mathbf{A} \odot \mathbf{B} = \begin{bmatrix} 1 \cdot 5 & 2 \cdot 6 \\ 3 \cdot 7 & 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.multiply(A, B)
print(C)

# yoki
C = A * B
print(C)
```

2.3 Matritsa ustida boshqa amallar
-------------

Matritsalarni ustida boshqa amallar ham bajarish mumkin. Ularning ba'zilari quyidagilardir:
- matritsa transponatsiyasi
- matritsa izi (trace)

### 2.3.1 Matritsa transponatsiyasi
Matritsa transponatsiyasi - bu matritsaning ustunlarini qatorlariga, qatorlarini esa ustunlariga almashtirish amaliyotidir.

Misol, agar $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$ bo'lsa, uning transponatasi quyidagicha bo'ladi:

$$\mathbf{A}^T = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = A.T
print(B)

# yoki
B = np.transpose(A)
print(B)

# yoki
B = A.transpose()
print(B)
```

### 2.3.2 Matritsa izi (trace)

Matritsa izi - bu matritsaning diagonal elementlarini qo'shish amaliyotidir.

$$\text{trace}(\mathbf{A}) = \sum_{i=1}^{n} a_{ii} = a_{11} + a_{22} + \ldots + a_{nn}$$

Misol, agar $$\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$ bo'lsa, uning izi quyidagicha bo'ladi:

$$\text{trace}(\mathbf{A}) = 1 + 4 = 5$$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
t = np.trace(A)
print(t)

# yoki
t = A.trace()
print(t)
```
