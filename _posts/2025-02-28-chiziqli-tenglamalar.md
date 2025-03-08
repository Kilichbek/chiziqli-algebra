---
title: Modul 3. Transformatsiya va Chiziqli Tenglamalar
author: Qilichbek Haydarov
date: 2025-02-28
category: Jekyll
layout: post
---

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/z2sn40AtLCE?si=1Mv43lAjin3cNAfn" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</center>

> ##### Birlik matritsa
>
> Diagonal elementlari birga teng bo'lgan (boshqa elementlari nol) kvadrat matritsa.
$$\textit{I}_n := \begin{bmatrix} 1 & 0 & \cdots & 0 & 0 \\ 0 & 1 & \cdots & 0 & 0 \\ 0 & 0 & \ddots & 0 & 0 \\ 0 & 0 & \cdots & 1 & 0 \\ 0 & 0 & \cdots & 0 & 1 \end{bmatrix} \in \mathbb{R}^{n \times n}$$
{: .block-tip }
Birlik matritsaga misollar:
$$\ I_{2}={\begin{bmatrix}1&0\\0&1\end{bmatrix}},\ I_{3}={\begin{bmatrix}1&0&0\\0&1&0\\0&0&1\end{bmatrix}},\ \dots$$

```python
import numpy as np

# Birlik Matritsa yaratish
np.eye(3)
```

> ##### Teskari matritsa
>
> Bizda $\mathbf{A} \in \mathbb{R}^{n \times n}$ matritsa bor. Unga teskari matritsa $\mathbf{A}^{-1}$ hisoblanadi, agar unda ushbu hususiyat bo'lsa:
> 
> $$\boxed {\mathbf{A} \mathbf{A}^{-1} = \mathbf{A}^{-1} \mathbf{A} = \mathbf{I}}$$
{: .block-tip }


```python
import numpy as np

# Teskari Matritsani hisoblash
A = np.array([[-1, 1.5 ], [1, -1]])
A_inv = np.linalg.inv(A)
A_inv
```

3.1 Chiziqli Transformatsiya
-------------

*Transformatsiya* - bu funksiya, biror vektorni oladi va boshqa vektor qaytaradi: 

$$T(\mathbf{v}) = A\mathbf{v}$$

Transformatsiya **chiziqli** hisoblanadi agar, ushbu shartlar bajarilsa:

- $T(\textbf{v} + \textbf{w}) = T(\textbf{v}) + T(\textbf{w})$
- $T(c\textbf{v}) = cT(\textbf{v})$


$f(x,y)=(2x,y)$ | ${\textstyle f(\mathbf {a} +\mathbf {b} )=f(\mathbf {a} )+f(\mathbf {b} )}$ | ${\textstyle f(\lambda \mathbf {a} )=\lambda f(\mathbf {a} )}$
- | - | -
![alt](https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Streckung_eines_Vektors.gif/311px-Streckung_eines_Vektors.gif) | ![alt](https://upload.wikimedia.org/wikipedia/commons/2/25/Streckung_der_Summe_zweier_Vektoren.gif) | ![alt](https://upload.wikimedia.org/wikipedia/commons/e/e6/Streckung_homogenitaet_Version_3.gif) 


### 3.1.1 Chiziqli Transformatsiyaga Misollar
- | - | - | - | -
Siljish (Shear) | Aks (Reflection) | Siqish (Squeeze) | Masshtablash (Scaling) | Aylanish (Rotation) 
$${\displaystyle {\begin{bmatrix}1& m\\0&1\end{bmatrix}}}$$ | $${\displaystyle {\begin{bmatrix}-1&0\\0&1\end{bmatrix}}}$$ | $${\displaystyle {\begin{bmatrix}{k}&0\\0&{\frac {1}{k}}\end{bmatrix}}}$$ | $${\displaystyle {\begin{bmatrix}{m}&0\\0&{m}\end{bmatrix}}}$$|${\displaystyle {\begin{bmatrix}\cos (\theta)&-\sin (\theta)\\\sin (\theta)&\cos (\theta)\end{bmatrix}}}$
![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/VerticalShear_m%3D1.25.svg/350px-VerticalShear_m%3D1.25.svg.png) | ![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Flip_map.svg/300px-Flip_map.svg.png) | ![](https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Squeeze_r%3D1.5.svg/300px-Squeeze_r%3D1.5.svg.png) | ![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Scaling_by_1.5.svg/250px-Scaling_by_1.5.svg.png) | ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Rotation_by_pi_over_6.svg/250px-Rotation_by_pi_over_6.svg.png) 

### 3.1.2 Affin Transformatsiya (Affine Mapping)
Affin transformatsiya 2 ta funksiyadan iborat: chiziqli transformatsiya + surish

$$\boxed{\displaystyle \mathbf {y} =f(\mathbf {x} )=\mathbf {A}\mathbf {x} +\mathbf {b}}$$

- $\mathbf {A}$ - chiziqli transformatsiya matritsasi
- $\mathbf{b}$ - surish vektori

Matritsa shaklida yozmoqchi bo'lsak, ushbu ko'rinishda yozishimiz mumkin:

$${\displaystyle {\begin{bmatrix}\mathbf {y} \\1\end{bmatrix}}=\left[{\begin{array}{ccc|c}&\mathbf {A}&&\mathbf {b} \\0&\cdots &0&1\end{array}}\right]{\begin{bmatrix}\mathbf {x} \\1\end{bmatrix}}}$$


3.2 Chiziqli tenglamalardan iborat tizim
-------------
Chiziqli tenglamalar tizimi quyidagi ko'rinishda ifodalanadi:

$${\displaystyle {\begin{cases}a_{11}x_{1}+a_{12}x_{2}+\dots +a_{1n}x_{n}=b_{1}\\a_{21}x_{1}+a_{22}x_{2}+\dots +a_{2n}x_{n}=b_{2}\\\vdots \\a_{m1}x_{1}+a_{m2}x_{2}+\dots +a_{mn}x_{n}=b_{m},\end{cases}}}$$

- ${\displaystyle x_{1},x_{2},\dots ,x_{n}}$ noma'lumlar
- ${\displaystyle a_{11},a_{12},\dots ,a_{mn}}$ koeffitsientlar
- ${\displaystyle b_{1},b_{2},\dots ,b_{m}}$ konstant sonlar

Matritsa ko'rinishida biz ularni quyidagicha yozsak bo'ladi:

$$\boxed{\mathbf {A}\mathbf{x} = \mathbf{b}}$$

bu yerda $\mathbf{A}$ - matritsa, $\mathbf{x}$ - noma'lum vektor, $\mathbf{b}$ - javob vektori:

$${\displaystyle \mathbf {A}={\begin{bmatrix}a_{11}&a_{12}&\cdots &a_{1n}\\a_{21}&a_{22}&\cdots &a_{2n}\\\vdots &\vdots &\ddots &\vdots \\a_{m1}&a_{m2}&\cdots &a_{mn}\end{bmatrix}},\quad \mathbf {x} ={\begin{bmatrix}x_{1}\\x_{2}\\\vdots \\x_{n}\end{bmatrix}},\quad \mathbf {b} ={\begin{bmatrix}b_{1}\\b_{2}\\\vdots \\b_{m}\end{bmatrix}}.}$$


### 3.2.1 Yechish 1: O'zgaruvchilarni yo'q qilish 

Chiziqli tenglamalar tizimini yechishning eng oddiy usuli *o'zgaruvchilarni qayta-qayta yo'q qilish*dir. Ushbu usulni quyidagicha ta'riflash mumkin:

1. Birinchi tenglamada o'zgaruvchilardan birini boshqalari bilan ifodalanadi.
2. Ushbu ifodani qolgan tenglamalarga qo'yiladi. Bu bitta kamroq tenglamali va noma'lum tenglamalar tizimini beradi.
3. Tizim bitta chiziqli tenglamaga keltirilguncha 1 va 2-bosqichlar takrorlanadi.
4. Oxiri hosil bo'lgan tenglama yechiladi, so'ngra butun yechim topilguncha chiqqan natijalar noma'lumlar o'rniga qo'yib boriladi.

Masalan, quyidagi chiziqli tenglamalar tizimini yechamiz:

$$
{\displaystyle {\begin{cases}x+3y-2z=5\\3x+5y+6z=7\\2x+4y+3z=8\end{cases}}}
$$

Bu yerda, biz $x$ noma'lumini boshqa noma'lumlar bilan ifodalaymiz. Misol uchun, birinchi tenglamadan $x$ ni ajratib olishimiz mumkin:
$${\displaystyle x=5+2z-3y}$$
va ikkinchi va uchinchi tenglamalarga $x$ o'rniga olgan ifodani kiritamiz:

$${\displaystyle {\begin{cases}y=3z+2\\y={\tfrac {7}{2}}z+1\end{cases}}}$$

Shunda bizda:

$${\displaystyle {\begin{aligned}3z+2={\tfrac {7}{2}}z+1\\\Rightarrow z=2\end{aligned}}}$$

va javob $${\displaystyle (x,y,z)=(-15,8,2)}$$ bo'ladi.

### 3.2.2 Yechish 2: Gauss usuli

**Gauss usuli** - chiziqli tenglamalar tizimini yechishning klassik usuli hisoblanadi. Nemis matematigi Karl Fridrix Gauss sharafiga nomlangan. Bu o'zgaruvchilarni ketma-ket yo'q qilish usuli bo'lib, elementar transformatsiyalar yordamida tenglamalar tizimi ekvivalent uchburchak tizimga tushiriladi. So'ngra, tizimning barcha o'zgaruvchilari oxiridan boshlab ketma-ket topiladi.

Misol, bizga ushbu chiziqli tenglamalar tizimi berilgan:

$$
{\displaystyle {\begin{cases}2x+y-z=8\\-3x-y+2z=-11\\-2x+y+2z=-3\end{cases}}}
$$

Bu sistemadan koeffitsient matritsa $\mathbf{A}$, noma'lumlar vektori $\mathbf{x}$ va javob vektori $\mathbf{b}$ ni ajratib olamiz:

$$\mathbf{A} =
\begin{bmatrix}
2 & 1 & -1\\
-3 & -1 & 2\\
-2 & 1  & 2
\end{bmatrix}$$, $$\mathbf{x} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$, $$\mathbf{b} = \begin{bmatrix} 8 \\ -11 \\ -3 \end{bmatrix}$$.

Shunda, chiziqli tenglamalar tizimi quyidagi ko'rinishga keltiriladi:

$$ \boxed{\mathbf{A}\mathbf{x} = \mathbf{b}}$$

Endi, Gauss usuli bo'yicha biz birinchi $\mathbf{A}$ matritsani $\mathbf{b}$ vektori bilan kengaytirib olamiz:


$$\begin{bmatrix} \mathbf{A} & | & \mathbf{b}  \end{bmatrix} $$

Ya'ni, quyidagi ko'rinishda:
$${\displaystyle \left[{\begin{array}{rrr|r}2&1&-1&8\\-3&-1&2&-11\\-2&1&2&-3\end{array}}\right]}$$

Shu yerda biz qator ustida amallardan foydalanib kengaytirilgan $\mathbf{A}$ matritsani yuqoridan uchburchak matritsaga o'tkazamiz. Amallar quyidagicha:

1. Ikki qatorning o'rnini almashtirish.
2. Qatorni nolga teng bo'lmagan songa ko'paytirish.
3. Bitta qatorga boshqa qatorning skalyar ko'paytmasini qo'shish.

Tepadagi misolni yechsak:

 Tenglamalar | Qator ustida amallar| Kengaytirilgan Matritsa 
 - | - | - 
 $$\begin{alignedat}{4}2x&{}+{}&y&{}-{}&z&{}={}&8&\\-3x&{}-{}&y&{}+{}&2z&{}={}&-11&\\-2x&{}+{}&y&{}+{}&2z&{}={}&-3&\end{alignedat}$$| |$$ \left[{\begin{array}{rrrr}2&1&-1&8\\-3&-1&2&-11\\-2&1&2&-3\end{array}}\right]$$ 
 $$\begin{alignedat}{4}2x&{}+{}&y&{}-{}&z&{}={}&8&\\&&{\tfrac {1}{2}}y&{}+{}&{\tfrac {1}{2}}z&{}={}&1&\\&&2y&{}+{}&z&{}={}&5&\end{alignedat}$$ | $${\displaystyle {\begin{aligned}Q_{2}+{\tfrac {3}{2}}Q_{1}&\to Q_{2}\\Q_{3}+Q_{1}&\to Q_{3}\end{aligned}}}$$ | $${\displaystyle \left[{\begin{array}{rrrr}2&1&-1&8\\0&{\frac {1}{2}}&{\frac {1}{2}}&1\\0&2&1&5\end{array}}\right]}$$ 
 $${\displaystyle {\begin{alignedat}{4}2x&{}+{}&y&{}-{}&z&{}={}&8&\\&&{\tfrac {1}{2}}y&{}+{}&{\tfrac {1}{2}}z&{}={}&1&\\&&&&-z&{}={}&1&\end{alignedat}}}$$ | $${\displaystyle Q_{3}+-4Q_{2}\to Q_{3}}$$ | $${\displaystyle \left[{\begin{array}{rrrr}2&1&-1&8\\0&{\frac {1}{2}}&{\frac {1}{2}}&1\\0&0&-1&1\end{array}}\right]}$$ 
 $${\displaystyle {\begin{alignedat}{4}2x&{}+{}&y&&&{}={}7&\\&&{\tfrac {1}{2}}y&&&{}={}{\tfrac {3}{2}}&\\&&&{}-{}&z&{}={}1&\end{alignedat}}} $$| $${\displaystyle {\begin{aligned}Q_{1}-Q_{3}&\to Q_{1}\\Q_{2}+{\tfrac {1}{2}}Q_{3}&\to Q_{2}\end{aligned}}}$$ |$${\displaystyle \left[{\begin{array}{rrrr}2&1&0&7\\0&{\frac {1}{2}}&0&{\frac {3}{2}}\\0&0&-1&1\end{array}}\right]}$$ 
 $${\displaystyle {\begin{alignedat}{4}2x&{}+{}&y&\quad &&{}={}&7&\\&&y&\quad &&{}={}&3&\\&&&\quad &z&{}={}&-1&\end{alignedat}}}$$ | $${\displaystyle {\begin{aligned}2Q_{2}&\to Q_{2}\\-Q_{3}&\to Q_{3}\end{aligned}}}$$ | $${\displaystyle \left[{\begin{array}{rrrr}2&1&0&7\\0&1&0&3\\0&0&1&-1\end{array}}\right]}$$ 
 $${\displaystyle {\begin{alignedat}{4}x&\quad &&\quad &&{}={}&2&\\&\quad &y&\quad &&{}={}&3&\\&\quad &&\quad &z&{}={}&-1&\end{alignedat}}}$$ | $${\displaystyle {\begin{aligned}Q_{1}-Q_{2}&\to Q_{1}\\ {\tfrac {1}{2}}Q_{1}&\to Q_{1}\end{aligned}}}$$ | $${\displaystyle \left[{\begin{array}{rrrr}1&0&0&2\\0&1&0&3\\0&0&1&-1\end{array}}\right]}$$ 

 va javob $${\displaystyle (x,y,z)=(2,3,-1)}$$ bo'ladi.

### 3.2.3 Yechish 3: Teskari matritsa yordamida

Agar biz $\mathbf{A}$ kvadrat matritsaga teskari matritsa $\mathbf{A}^{-1}$ ni topa olsak, chiziqli tenglamalar tizimini quyidagi ko'rinishda yechishimiz mumkin:

$$\mathbf{A}\mathbf{x} =\mathbf{b},$$

$$\mathbf{A}^{-1}\mathbf{A}\bf{x} = \mathbf{A}^{-1}\bf{b},$$

$$\mathbf{I}\bf{x} = \mathbf{A}^{-1}\bf{b},$$

$$\boxed{\bf{x} = \mathbf{A}^{-1}\bf{b}}$$

Ya'ni, shunchaki $\mathbf{A}^{-1}$ matritsasini topib, $\mathbf{b}$ vektoriga ko'paytirib yechishimiz mumkin.

```python
import numpy as np

# Chiziqli tenglamalar tizimini yechish
A = np.array([
    [2, 1, -1], 
    [-3, -1, 2], 
    [-2, 1, 2]
])
b = np.array([8, -11, -3])

# Teskari matritsa yordamida yechish
x = np.linalg.inv(A).dot(b)
x
```
