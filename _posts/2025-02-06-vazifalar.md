---
title: Vazifalar
author: Qilichbek Haydarov
date: 2025-01-13
category: Jekyll
layout: post
---


Vazifa 1: Vektorlar va Vektor Algebra
-------------

### 1.1 Vektor ustida amallar


Agar $$\textbf{u}= \begin{bmatrix} 5 \\ 2 \\ 9 \end{bmatrix}$$ va $$\textbf{v}= \begin{bmatrix} 1 \\ -8 \\ 6 \end{bmatrix}$$, $c=0.5$ bo'lsa, quyidagilarni bajaring:

1. $\textbf{u} + \textbf{v}$
2. $\textbf{u} - \textbf{v}$
3. $c\textbf{u}$
4. $\textbf{v}$ uzunligi 
5. $\textbf{u}$ larining birlik vektori ko'rinishi

### 1.2 Vektor normasi va skalyar ko'paytma

Agar $$\textbf{u}= \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}$$ va $$\textbf{v}= \begin{bmatrix} 3 \\ -1 \\ 4 \end{bmatrix}$$, quyidagilarni toping:

1. $\textbf{u}$ vektorining $L_1$ normasi
2. $\textbf{v}$ vektorining $L_2$ normasi
3. $\textbf{u} \cdot \textbf{v}$
4. $d(\textbf{u}, \textbf{v})$ (Evklid masofa)
5. $\textbf{u}$ va $\textbf{v}$ orasidagi burchak

### 1.3 Chiziqli kombinatsiya va bog'liqlik

Quyidagilarni aniqlang:

1. $$\textbf{u}= \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \textbf{v}= \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \textbf{w}= \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$ chiziqli bog'liqmi?

2. $$\textbf{u}= \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \textbf{v}= \begin{bmatrix} 3 \\ 4 \end{bmatrix}, \textbf{w}= \begin{bmatrix} 8 \\ 14 \end{bmatrix}$$ chiziqli bog'liqmi?


### 1.4 📌 Mini-Loyiha: Oddiy Film Tavsiya Qiluvchi Tizim

![Netflix](https://img.shields.io/badge/Netflix-E50914?style=for-the-badge&logo=netflix&logoColor=white)
Tasavvur qiling siz Netflix  companiyasida ishlayapsiz. Ma'lumotlar bazasi:

| Film nomi         | Jangari ⚔️ | Romantik 💕 | Fantastika 🚀 | Drama 🎭 | Komediya 😂 |
|------------------|----------|-----------|--------------|----------|------------|
| **Sheryurak**    | 5        | 1         | 2            | 1        | 0          |
| **Titanik**      | 1        | 4         | 2            | 2        | 1          |
| **Avatar**       | 3        | 2         | 5            | 2        | 1          |
| **Buyuk Getsbi** | 1        | 2         | 3            | 4        | 0          |
| **Pushti Pantera** | 1      | 1         | 4            | 1        | 5          |

**Foydalanuvchi qiziqishlari**:
```python
[Jangari = 5, Romantik = 3, Fantastika = 1, Drama = 2, Komediya = 2]
```

🎯 **Maqsad**: Foydalanuvchining yoqtirgan kino janrlariga asoslanib, unga mos filmlarni tavsiya qilish.

-------

![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

> ##### Google Colab da vazifani bajarishingiz mumkin!
>
> [**Vazifa 1: Vektorlar va Vektor ustida amallar**](https://colab.research.google.com/drive/1GTfQglVxzgAF2h5aPRuy_DD7g8YKZt_a?usp=sharing)
{: .block-warning }

-------

Vazifa 2: Matritsa Asoslari
-------------

### 2.1.1 Matritsa yaratish
Quyidagi berilgan matritsa elementlarini shunda o'zgartiringki, natijada berilga kulgan smaylik rasmi chiqsin.
Izoh: siz pikselni o'zgartirishda, matritsa elementlarini o'zgartirish orqali rasmda o'zgarishni ko'rsatishingiz mumkin.
Buning uchun 0 va 1 sonlaridan foydalanishingiz mumkin. 0 - oq rang, 1 - qora rang.

```python
import numpy as np
import matplotlib.pyplot as plt

# < sizning kodlaringiz bu yerda yozing>
smile = np.zeros((16, 16), dtype=int)

# <pastdagi matritsa elementlarini o'zgartirng>
smile[4:6, 4:6] = 1
smile[4:6, 10:12] = 1
smile[10, 4:12] = 1
smile[11, 3:5] = 1
smile[11, 11:13] = 1


# Yaratilgan matritsa shunday ko'rinishda bo'lish kerak
plt.imshow(smile, cmap='gray')
```

### 2.1.2 Matritsa Yaratish

Quyidagi funksiyalardan foydalanib turli xil matritsalarni yarating:
```python
np.random.randint(a, b, (m, n)) # a va b orasidagi tasodifiy sonlar bilan to'ldirilgan m x n matritsa
np.zeros((m, n)) # nol elementlar bilan to'ldirilgan m x n matritsa
np.ones((m, n)) # bir elementlar bilan to'ldirilgan m x n matritsa
```

1. 5 x 100 elementlarini 0 va 50 orasidagi tasodifiy sonlar bilan to'ldirilgan matritsa
2. 100 x 5 nol elementlar bilan to'ldirilgan matritsa
3. 5 x 5 bir elementlar bilan to'ldirilgan matritsa

### 2.2 Matritsa Ustida Amallar

Quyidagi matritsalarni yaratib, ular ustida amallar bajarib ko'ring:

1. $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$ matritsalarni yarating.
2. $A + B$ va $A - B$ amallarini toping
3. $A B$ amalini toping
4. $A \textbf{v}$ amalini toping, $\textbf{v} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
5. $A B \textbf{v}$ amalini toping
6. $A \odot B $ amalini toping
7. $A^T$ va $B^T$ amallarini toping
8. $A \odot B^T$ amalini toping
9. $A^T B$ amalini toping
10. $A^T B^T$ amalini toping
11. $\text{tr}(A)$ va $\text{tr}(B)$ amallarini toping
12. $\text{tr}(A B)$ va $\text{tr}(B A)$ amallarini toping
13. $\text{tr}(A B^T)$ va $\text{tr}(B^T A)$ amallarini toping
14. $\text{tr(A - B)}$ va $\text{tr(A) - tr(B)}$ tengliklarini tekshiring


### 2.3 🎨 **Mini-Loyiha: Tasvirni Manipulyatsiya Qilish** 🖼️

**🎯 Maqsad**  
Berilgan tasvir ustida **matritsa amallari** yordamida ishlov berish va uni **kulrang tasvirga (grayscale)** aylantirish.

---

**📌 Tasvir va Matritsa Tushunchasi**  
Rangli tasvir (RGB) **3 ta matritsadan** iborat:  
🔴 **R (Red)** – Qizil rang matritsasi  
🟢 **G (Green)** – Yashil rang matritsasi  
🔵 **B (Blue)** – Ko‘k rang matritsasi  

Har bir matritsa **0 dan 255 gacha** bo‘lgan qiymatlardan iborat bo‘lib, tasvirning har bir pikselini ifodalaydi.  
**Misol:** Quyidagi rangli tasvirni **uchta matritsaga** ajratish mumkin:

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Rgb-compose-Alim_Khan.jpg/2560px-Rgb-compose-Alim_Khan.jpg" width=600>

📷 **Tasvir haqida ma’lumot**:  
Bu rasm **Buxoro amiri Muhammad Olimxon (1880–1944)** ga tegishli bo‘lib, 1911-yilda Sergey Prokudin-Gorskiy tomonidan olingan.  
Surat uch xil **ko‘k, yashil va qizil (RGB)** filtrlar yordamida tushirilgan va keyinchalik rangli tasvirga aylantirilgan.

---
**📌 Kulrang Tasvir (Grayscale)**
Kulrang tasvirda faqat **yorug‘lik intensivligi** saqlanadi, ranglar esa yo‘qoladi.  
Tasvirni kulrangga aylantirish uchun quyidagi formuladan foydalanamiz:  

$$
Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B
$$

Bu yerda:  
✅ **Y** – Yorug‘lik intensivligi matritsasi (grayscale)  
✅ **R, G, B** – Qizil, yashil va ko‘k ranglar matritsalari  

---

**📌 Qadamlar**
🛠️ **1. Tasvirning maksimum va minimum piksel qiymatlarini topish**  
   - `np.max()` va `np.min()` funksiyalaridan foydalaning.  

📏 **2. Tasvir o‘lchamini aniqlash**  
   - Tasvir matritsasining **qator, ustun va kanallar sonini** toping.  

🌈 **3. Tasvirni RGB kanallariga ajratish**  
   - Har bir rang matritsasini ajratish uchun `img[:, :, i]` indekslashdan foydalaning.  

📊 **4. Har bir rang matritsasini [0,1] oralig‘iga normallashtirish**  
   - **Formula**:  
    $$
     I = \frac{X - X_{min}}{X_{max} - X_{min}}
    $$

⚫ **5. Tasvirni "Grayscale" ga o‘tkazish**  
   - **Formula**:  
     $$
     Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B
     $$
   - O‘lchamni o‘zgartirmasdan, ranglarni intensivlikka aylantiring.  

---

🎯 **Natija**: Siz tasvirni **matritsalar asosida tahlil qilish** va uni **kulrang formatga o‘tkazish** bo‘yicha bilim va tajriba olasiz! 🚀 

-------

![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9A825.svg?style=for-the-badge&logo=googlecolab&logoColor=white)

> ##### Google Colab da vazifani bajarishingiz mumkin!
>
> [**Vazifa 2: Matritsa Asoslari**](https://colab.research.google.com/drive/1Ql_fE5xR_Lys8y9Wq3DYCFTH0Hr_sKOU?usp=sharing)
{: .block-warning }
