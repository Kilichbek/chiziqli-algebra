---
title: Modul 5. Il'gor matritsa tushunchalari
author: Qilichbek Haydarov
date: 2025-02-28
category: Jekyll
layout: post
---

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/Y-J4fgV-iQ4?si=s5exf2dZ7Yc95q0A" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</center>

Bu darsda biz matritsalar bilan bog'liq ba'zi muhim tushunchalarni ko'rib chiqamiz. Bu tushunchalar sizga matritsalar bilan ishlashda va ularni yanada chuqurroq tushunishga yordam beradi.



4.1 Determinant
-------------
### 4.1.1 Determinant nima?

Determinant - bu kvadrat matritsa bilan bog'liq bo'lgan son. U matritsaning o'ziga xos xossalarini ifodalaydi va ko'plab matematik va fizik muammolarni hal qilishda muhim rol o'ynaydi.
U $$\text{det}(\mathbf{A})$$ yoki $$|\mathbf{A}|$$ bilan ifodalanadi.

Determinantning asosiy xususiyatlari:
1. $\text{det}(\textbf{I}) = 1$
2. $\text{det}(\textbf{A}) = 0 \Rightarrow \textbf{A}$ matritsa teskari mavjud emas
2. $\text{det}(c \textbf{A}) = c^n \text{det}(\textbf{A})$
3. $\text{det}(\textbf{A}^T) = \text{det}(\textbf{A})$
4. $\text{det}(\textbf{A}^{-1}) = \frac{1}{\text{det}(\textbf{A})}$
5. $\text{det}(\textbf{AB}) = \text{det}(\textbf{A}) \text{det}(\textbf{B})$
6. $\text{det}(\textbf{A}^n) = (\text{det}(\textbf{A}))^n$
7. $\text{det}(\textbf{A}) = \prod_{i=1}^{n} a_{ii}~$, agar $\textbf{A}$ uchburchak matritsa bo'lsa


### 4.1.2 Determinantning Geometrik ma'nosi?

- 2-D'da determinant - bu ikki vektor hosil qilgan parallelogramning maydoni
- 3-D'da determinant - bu uchta vektor hosil qilgan parallelepipedning hajmi

| 2D parallelogram | 3D parallelepiped|
![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Area_parallelogram_as_determinant_modified.svg/500px-Area_parallelogram_as_determinant_modified.svg.png) | ![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Determinant_parallelepiped.svg/570px-Determinant_parallelepiped.svg.png) |
| $$\textbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$ | $$\textbf{A} = \begin{bmatrix} \mathbf{r_1} & \mathbf{r_2} & \mathbf{r_3} \end{bmatrix}$$|


### 4.1.3 Determinantni hisoblash?
1. $2 \times 2$ matritsa uchun:
$$\text{det}(\textbf{A}) = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$
2. $3 \times 3$ matritsa uchun:
$$\text{det}(\textbf{A}) = \begin{vmatrix} a & b & c \\ d & e & f \\ g & h & i \end{vmatrix} = aei + bfg + cdh - ceg - bdi - afh$$
3. $n > 3$ uchun: 
    - kofaktorlar yordamida
    - tayanch (pivot) formulasi yordamida 


```python
import numpy as np
A = np.array([
    [1, 2, 3], 
    [4, 5, 6], 
    [7, 8, 9]
])
det_A = np.linalg.det(A)
print(f'A matritsaning determinanti: {det_A}')
```

4.2 "Eigen" tushuncha
-------------

Ushbu animatsiyada, qizil chiziq bo'yicha xarakatlanayotgan vektorlar ko'rsatilgan. Bu vektorlar matritsa bilan ko'paytirilganda o'z yo'nalishini o'zgartirmaydi, balki faqat uzunligi o'zgaradi. Bunday vektorlar **eigenvektorlar** deb ataladi:
<br>
<center><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Eigenvectors_of_a_linear_operator.gif/400px-Eigenvectors_of_a_linear_operator.gif"/></center>

Yana bir misol:
Ushbu Mona Lisa rasmini transformatsiya (siljitish) qilganimizda, qizil ko'rsatkich yo'nalishini o'zgartirgan, ko'k ko'rsatgich esa o'z yo'nalishini o'zgartirmagan. Bunday ko'rsatgichlar **eigenvektorlar** deb ataladi. Ular transformatsiya ostida o'z yo'nalishini o'zgartirmaydi, balki faqat uzunligi o'zgaradi.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Mona_Lisa_eigenvector_grid.png/960px-Mona_Lisa_eigenvector_grid.png">

Matematik tarzda bunday vektorlar quyidagi tenglama orqali ifodalanadi:

$$\boxed{\textbf{A} \mathbf{v} = \lambda \mathbf{v}}$$

Bu erda $\textbf{A}$ - matritsa, $\mathbf{v}$ - eigenvektor va $\lambda$ - eigenqiymat (eigenvalue).

<center>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Eigenvalue_equation.svg/500px-Eigenvalue_equation.svg.png" width=400>
</center>



### 4.2.1 Eigenvektorlar va Eigenqiymatlarni topish
1. $\textbf{A} \mathbf{v} = \lambda \mathbf{v}$ tenglamasini quyidagi ko'rinishga keltiramiz:
$$\textbf{A} \mathbf{v} - \lambda \mathbf{v} = \mathbf{0}$$
2. So'ng quyidagi ko'rinishga keltiramiz:
$$ (\textbf{A}- \lambda \mathbf{I}) \mathbf{v} = \mathbf{0}$$
3. $\mathbf{v}$ vektorning nolga teng bo'lmagan yechimi bo'lishi uchun, faqat va faqat $\text{det}(\textbf{A}- \lambda \mathbf{I}) = 0$ bo'lishi kerak. Bu erda $\mathbf{I}$ - birlik matritsa. Bu tenglama **xarakteristik tenglama** deb ataladi.
4. Bu tenglamani yechish orqali $\lambda$ eigenqiymatlarni topamiz.
5. Har bir $\lambda$ uchun $ (\textbf{A}- \lambda \mathbf{I}) \mathbf{v} = \mathbf{0}$ tenglamasini yechish orqali $\mathbf{v}$ eigenvektorlarni topamiz.

Misol: 

Bizga $$\textbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$ matritsa berilgan. Maqsadimiz uning eigenvektor va eigenqiymatlarini topish.

Xarakteristik tenglama:

$$\det(A-\lambda I)={\begin{vmatrix}2-\lambda &1\\1&2-\lambda \end{vmatrix}}=3-4\lambda +\lambda ^{2}$$

$$ \lambda = 1 \text{ yoki } \lambda = 3$$

Shunda, eigenvektorlarni topish uchun quyidagi tenglamani yechamiz:

|$\lambda = 1$ | $\lambda = 3$|
|$$\mathbf{A}-\lambda \mathbf{I} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$| $$\mathbf{A}-\lambda \mathbf{I} = \begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}$$|
|$$\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}\textbf{v} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$| $$\begin{bmatrix} -1 & 1 \\ 1 & -1 \end{bmatrix}\textbf{v} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$|
|$$\mathbf {v}_{\lambda =1}={\begin{bmatrix}1\\-1\end{bmatrix}}$$|$$\mathbf {v} _{\lambda =3}={\begin{bmatrix}1\\1\end{bmatrix}}$$ |



### 4.2.1 Eigen xossalari va Eigen-vektor matritsasi

- ${\displaystyle \operatorname {tr} (A)=\sum _{i=1}^{n}a_{ii}=\sum _{i=1}^{n}\lambda _{i}=\lambda _{1}+\lambda _{2}+\cdots +\lambda _{n}.}$
- ${\displaystyle \det(A)=\prod _{i=1}^{n}\lambda _{i}=\lambda _{1}\lambda _{2}\cdots \lambda _{n}.}$
- $\textbf{A}$ matritsaning teskarisi mavjud, agar uning har bir eigenqiymatlari $0$ ga teng bo'lmasa. 
- Yuqoridan (quyidan) uchburchak matritsaning eigen-qiymatlari dioganal elementlarga teng

- bizda $n$ ta chiziqli bog'liq bo'lmagan eigen-vektorlar mavjud $\{ \textbf{s}_1, \textbf{s}_2, ..., \textbf{s}_n \}$

$$\textbf{S} = \Bigg[ \mathop{\mathbf s_1}\limits_|^| \ \mathop{\mathbf s_2}\limits_|^| \ \cdots \  \mathop{\mathbf s_n}\limits_|^| \Bigg]$$

- $\textbf{S}$ - eigen-vektor matritsasi

4.3 Matritsa normasi (Matrix norm)
-------------

### 4.3.1  Matritsa normasi tushunchasi va xossalari

Matritsa normasi - bu matritsaning "o'lcham"ini belgilovchi, manfiy bo'lmagan skalar qiymat qaytaruvchi funksiya. 


- $$\|A\| \geq 0$$
- $$\|\alpha A\| = |\alpha| \|A\|$$
- $$\|A + B\| \leq \|A\| + \|B\|$$
- $$\|AB\| \leq \|A\| \|B\|$$


### 4.3.2  Matritsa normasi turlari

1. Frobenius normasi (Frobenius norm)


1. Frobenius normasi (Frobenius norm)

$${\displaystyle \|A\|_{\text{F}}={\sqrt {\sum _{i}^{m}\sum _{j}^{n}|a_{ij}|^{2}}}}$$

2. 1-normasi (1-norm)

$${\displaystyle \|A\|_{1 }=\max _{j = 1, ..., n}\sum_{j=1}^{n}|a_{ij}|.}$$

3. Spektr normasi (Spectral norm)
$${\displaystyle \|A\|_{2}={\sqrt {\lambda _{\max }\left(A^{T}A\right)}}}$$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
frobenius_norm = np.linalg.norm(A, 'fro')
max_norm = np.linalg.norm(A, 1)
spectral_norm = np.linalg.norm(A, 2)
print(f'Frobenius norm: {frobenius_norm}')
print(f'Maximum norm: {max_norm}')
print(f'Spectral norm: {spectral_norm}')

```

4.4 Matritsani Yoyish (Matrix Decomposition)
-------------

Matritsani yoyish degani bu matritsani boshqa matritsalarga bo'lishdir. Bu jarayon ko'plab matematik va fizik muammolarni hal qilishda qo'llaniladi. Yoyish jarayoni matritsaning o'ziga xos xossalarini aniqlashga yordam beradi. Masalan siz $6 = 2 \cdot 3$ deb yozishingiz mumkin. Bu erda $2$ va $3$ - $6$ ning yoyishidir ya'ni faktorlarga ajratish. Faktorlarga ajratganimizda masalan biz $6$ sonining xususiyatlarni aniqlaymiz: $2$ va $3$ ga bo'linishi, just son ekanligi. 


Matritsani yoyishning ko'plab usullari mavjud. Qo'llanilishiga qarab, ushbu usullarning ba'zilari boshqalarga qaraganda foydaliroqdir. Bu darsda biz matritsani yoyishning eng ko'p qo'llaniladigan usullari bilan tanishamiz:
1. $\textbf{LU}$ yoyish
2. $\textbf{QR}$ yoyish
3. $\textit{Eigen}$ yoyish
4. $\textbf{SVD}$ yoyish

### 4.4.1 $\textbf{LU}$ yoyish

Matritsani yoyishning keng tarqalgan usullaridan biri $\textbf{LU}$ yoyishdir. Odatda chiziqli tenglamalar tizimini yechishda va matritsaning determinantini topish uchun foydalaniladi.

$\textbf{LU}$ yoyishi Gauss usulini osonroq bajarish uchun kvadrat matritsani ikkita uchburchak matritsalarga parchalaydi. U ushbu chiziqli algebra teoremasiga asoslanadi:

Har qanday singulyar (ya'ni teskarisi mavjud emas) bo'lmagan kvadrat matritsa $\textbf{A}$, bir xil tartibdagi ikkita uchburchak matritsaning $\textbf{L}$ va $\textbf{U}$ ko'paytmasi sifatida yozilishi mumkin, shunday qilib 

- $\textbf{L}$  matritsa quyidan uchburchak matritsa (asosiy diagonal ustidagi barcha elementlar 0 ga teng) 
- $\textbf{U}$  yuqoridan uchburchak matritsa (asosiy diagonal ostidagi barcha elementlar 0 ga teng).


Matematik ifodasi:


$$ \begin{split}
\underset{n\times n}{\mathbf{A}} &= \underset{n\times n}{\mathbf{L}}~ \underset{n\times n}{\mathbf{U}} \\[2ex]
&= \begin{bmatrix}
l_{11} & 0 & 0 & \ldots & 0\\
l_{21}& l_{22} & 0 & \ldots & 0\\
l_{31}& l_{32} & l_{33} & \ldots & 0\\
\vdots & \vdots &\vdots & \ddots & \vdots \\
l_{n1} & l_{n2} & l_{n3} & \ldots & l_{nn}
\end{bmatrix}\begin{bmatrix}
u_{11} & u_{12} & u_{13} & \ldots & u_{1n}\\
0 & u_{22} & u_{23} & \ldots & u_{2n}\\
0 & 0 & u_{33} & \ldots & u_{3n}\\
\vdots & \vdots  &\vdots & \ddots & \vdots \\
0 & 0 & 0 & \ldots & u_{nn}
\end{bmatrix}
\end{split}$$

#### 4.4.1.1 $\textbf{LU}$ yoyishni hisoblash

Maqsadimiz: bizga berilgan $\textbf{A}$ matritsasini $\textbf{LU}$ yoyish orqali $\textbf{L}$ va $\textbf{U}$ matritsalarini topish.

Deylik, misol uchun, bizga quyidagi matritsa berilgan:

$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 3 & 1 \\ -2 & 3 & -2 \end{bmatrix}$$

Bu matritsanin $\textbf{LU}$ yoyishini topish uchun quyidagi qadamlarni bajaramiz:

1. $\textbf{A}$ matritsani birlik matritsa bilan kengaytiramiz:


$$\overbrace{\left[ \begin{array}{rrr} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{array} \right]}^{\textbf{I}} \overbrace{ \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 2 & 3 & 1 \\ -2 & 3 & -2 \end{array} \right]\nonumber}^{\textbf{A}}$$

2. Qator ustida amallar orqali $\mathbf{A}$ matritsaning diagonalidan pastki elementlarni 0 ga aylantiramiz. Va har bir qator ustida amallar bajarish jarayonida $\textbf{I}$ matritsasini unga teskari qator ammalarni qo'llab boramiz.


$R_2 = R_2 - 2R_1$ : $$ \quad \left[ \begin{array}{rrr} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 0 & 0 & 1 \end{array} \right] \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 0 & -1 & -5 \\ -2 & 3 & -2 \end{array} \right]\nonumber$$


$R_3 = R_3 + 2 R_1$: $$ \quad \left[ \begin{array}{rrr} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -2 & 0 & 1 \end{array} \right] \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 0 & -1 & -5 \\ 0 & 7 & 4 \end{array} \right]\nonumber$$



$R_3 = R_3 + 7 R_2$: $$\quad \left[ \begin{array}{rrr} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -2 & -7 & 1 \end{array} \right] \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 0 & -1 & -5 \\ 0 & 0 & -31 \end{array} \right]\nonumber$$

$$\textbf{A} = \overbrace{\left[ \begin{array}{rrr} 1 & 0 & 0 \\ 2 & 1 & 0 \\ -2 & -7 & 1 \end{array} \right]}^{\textbf{L}}  \overbrace{ \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 0 & -1 & -5 \\ 0 & 0 & -31 \end{array} \right] }^{\textbf{U}}\nonumber$$

#### 4.4.1.2 LU Yoyish usuli orqali $\textit{A} \mathbf{x} =\mathbf{b}$ yechish
Agar biz $\textbf{A} \mathbf{x} = \mathbf{b}$ tenglamalar tizimini yechishni istasak, biz $\textbf{A} = \textbf{LU}$ ni topishimiz kerak.

$$ \begin{aligned}
{\bf A x} &= {\bf b} \\
{\bf L U x} &= {\bf b} \\
{\bf U x} &= {\bf L}^{-1} {\bf b} \\
{\bf x} &= {\bf U}^{-1} ({\bf L}^{-1} {\bf b}),
\end{aligned}$$

- $\textbf{L}^{-1}\mathbf{b}$ - to'g'ri o'rniga qo'yish (forward substitution)
- $\textbf{U}^{-1}(\textbf{L}^{-1}\textbf{b})$ - teskari o'rniga qo'yish (backward substitution)

### 4.4.2 $\textbf{QR}$ yoyish

$\textbf{QR}$ yoyish - bu matritsaning $\textbf{Q}$ ortogonal matritsa va $\textbf{R}$ yuqoridan uchburchak matritsa ko'rinishida yozilishi.

$$\boxed{\textbf{A} = \textbf{QR}}$$

$\textbf{QR}$ yoyishni topish uchun
1. Gram-Shmidt jarayonini qo'llash va $\textbf{Q}$ matritsasini topish
2. $\textbf{R}$ matritsasini topish
$$\textbf{A} = \textbf{QR} = \Bigg[ \mathop{\mathbf e_1} \limits_|^| \ \mathop{\mathbf e_2} \limits_|^| \ \cdots \ \mathop{\mathbf e_n} \limits_|^| \Bigg] \begin{bmatrix} \mathbf{e}_1 \cdot \mathbf{a}_1 & \mathbf{e}_1 \cdot \mathbf{a}_2 & \cdots & \mathbf{e}_1 \cdot \mathbf{a}_n \\ 0 & \mathbf{e}_2 \cdot \mathbf{a}_2 & \cdots & \mathbf{e}_2 \cdot \mathbf{a}_n \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \mathbf{e}_n \cdot \mathbf{a}_n \end{bmatrix}$$

Misol: $${\displaystyle A={\begin{bmatrix}1&1&1\\0 & 1 & 1\\1&1&0\end{bmatrix}} }$$.

$$\textbf{Q}={\begin{bmatrix}{\frac {\mathbf {u}_{1}}{\|\mathbf {u}_{1}\|}}&{\frac {\mathbf {u}_{2}}{\|\mathbf {u}_{2}\|}}&{\frac {\mathbf {u}_{3}}{\|\mathbf {u}_{3}\|}}\end{bmatrix}} = \begin{split}
\begin{bmatrix}
1/\sqrt{2} & 0 & 1/\sqrt{2} \\
0 & 1 & 0 \\
1/\sqrt{2} & 0 & -1/\sqrt{2}
\end{bmatrix}
\end{split}$$

$$\textbf{A} = \textbf{QR} \Rightarrow \textbf{Q}^{\textsf {T}}\textbf{A}=\textbf{Q}^{\textsf {T}}\textbf{Q}\,\textbf{R}=\textbf{R};$$

$${\displaystyle {\begin{aligned}\\\textbf{R}&=\textbf{Q}^{\textsf {T}}\textbf{A}={\begin{bmatrix}
\sqrt{2} & \sqrt{2} & 1/\sqrt{2} \\
0 & 1 & 1 \\
0 & 0 & 1/\sqrt{2}
\end{bmatrix}}.\end{aligned}}}$$

```python
import numpy as np
A = np.array([[1, 1, 1],
              [0, 1, 1],
              [1, 1, 0]])

Q, R = np.linalg.qr(A)
print(Q)
print(R) 
```

### 4.4.3 $\textit{Eigen}$ yoyish

Eigen yoyish - bu $n \times n$ kvadrat matritsaning eigen-qiymatlar va eigen-vektorlar bilan yoyish usuli.

$$\boxed{\displaystyle \mathbf {A} =\mathbf {X} \mathbf {\Lambda } \mathbf {X} ^{-1}}$$

- $\mathbf {X}$ - ustunlari eigen-vektorlardan hosil bo'lgan matritsa
- $\mathbf {\Lambda }$ - diagonal matritsa, diagonal elementlari eigen-qiymatlar


Eigen yoyishni topish uchun:
1. Eigen-qiymatlar $\lambda_1, \lambda_2, ..., \lambda_n$ va eigen-vektorlar $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n$ topiladi
2. $\mathbf {X}$ va $\mathbf {\Lambda }$ matritsalarni topish

$$\textbf{A} = \textbf{X} \mathbf{\Lambda} \textbf{X}^{-1} = \Bigg[ \mathop{\mathbf v_1} \limits_|^| \ \mathop{\mathbf v_2} \limits_|^| \ \cdots \ \mathop{\mathbf v_n} \limits_|^| \Bigg] \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix}\Bigg[ \mathop{\mathbf v_1} \limits_|^| \ \mathop{\mathbf v_2} \limits_|^| \ \cdots \ \mathop{\mathbf v_n} \limits_|^| \Bigg] ^{-1}$$

Qulaylik:
- $\mathbf {A }^k = \mathbf {X} \mathbf {\Lambda }^k \mathbf {X} ^{-1}$

Misol: $${\displaystyle \mathbf {A} ={\begin{bmatrix}1&2\\2&1\end{bmatrix}}}$$. Uning eigen yoyilishini toping.

1. Eigen-qiymatlar: $\lambda_1 = 3, \lambda_2 = -1$ va eigen-vektorlar: $$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$
2. $${\displaystyle \mathbf {X} ={\begin{bmatrix}1&-1\\1&1\end{bmatrix}}}$$ va $${\displaystyle \mathbf {\Lambda }={\begin{bmatrix}3&0\\0&-1\end{bmatrix}}}$$


### 4.4.4 $\textbf{SVD}$ yoyish

$$\boxed{\displaystyle \mathbf {A} =\mathbf {U\Sigma V^{T}}}$$

Eigen yoyish faqat $n \times n$ matritsalarda ishlaydi. Singulyar yoyish esa $m \times n$ matritsalar uchun ishlaydi.

Singulyar qiymat - bu $\textbf{A}^T \textbf{A}$ va $\textbf{A} \textbf{A}^T$ matritsalarning eigen-qiymatlari.

- $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_r \geq 0$
- $\sigma_i = \sqrt{\lambda_i}$ - singulyar qiymatlar

    
$$\boxed{\displaystyle \mathbf {A} =\mathbf {U\Sigma V^{T}}}$$

- $\mathbf {U}$ - $m \times m$ ortogonal matritsa: ustunlari $\textbf{A}\textbf{A}^T$ matritsaning eigen-vektorlaridan tashkil topkan
- $\mathbf {V}$ - $n \times n$ ortogonal matritsa: ustunlari $\textbf{A}^T \textbf{A}$ matritsaning eigen-vektorlaridan tashkil topkan
- $\mathbf {\Sigma}$ - $m \times n$ diagonal elementlari singulyar qiymatlar iborat diagonal matritsa 

$$\boxed{\displaystyle \mathbf {A} =\mathbf {U\Sigma V^{T}}}$$

$$\textbf{A} = \mathbf {U\Sigma V^{T}} = \Bigg[ \mathop{\mathbf u_1} \limits_|^| \ \mathop{\mathbf u_2} \limits_|^| \ \cdots \ \mathop{\mathbf u_m} \limits_|^| \Bigg] \left[\begin{array}{ccc|c}
\sigma_1 & & & \\
& \ddots & & \boldsymbol{0} \\
& & \sigma_r & \\ \hline
& \boldsymbol{0} & & \boldsymbol{0}
\end{array} \right]_{m \times n} \Bigg[ \mathop{\mathbf v_1} \limits_|^| \ \mathop{\mathbf v_2} \limits_|^| \ \cdots \ \mathop{\mathbf v_n} \limits_|^| \Bigg] ^{T}$$

$\mathbf{u}_i$ va $\mathbf{v}_i$ orasidagi bog'liqlik:
- $\mathbf{A}^T \mathbf{u}_i = \sigma_i \mathbf{v}_i$
- $\mathbf{A} \mathbf{v}_i = \sigma_i \mathbf{u}_i$

<center>
    <img src="https://gregorygundersen.com/image/svd/standard.png" width=400>
</center>


Singulyar yoyishni topish uchun:
1. $\textbf{A}^T \textbf{A}$ eigen-qiymatlari va eigen-vektorlar topish
2. $\mathbf{\Sigma}$ matritsasini singulyar qiymatlari bilan to'ldirish
3. $\textbf{V}$ ni normallashtirilgan eigen-vektorlar bilan to'ldirish
4. $\textbf{u}_i = \frac{1}{\sigma_i}\textbf{A}\mathbf{v}_i$ orqali $r$ vektor toping. $\mathbb{R}^m$ fazoga Gram-Shmit yoki boshqa yo'l bilan $r$ vektorlarni $m$ ortonormal bazislarga kengaytiring, va $\textbf{U}$ matritsaning shakllantiring


Misol $${\displaystyle \mathbf {A} ={\begin{bmatrix} 1 & 1 \\ 1 & -1 \\ 0 & 1 \end{bmatrix}}}$$. Uning singulyar yoyilishini toping.
1. $\textbf{A}^T \textbf{A}$ = ${\displaystyle {\begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}}}$
    - eigen-qiymatlar: $\lambda_1 \geq \lambda_2$, $\lambda_1 = 3, \lambda_2 = 2$
    - eigen-vektorlar (normallashtirilgan): $\mathbf{v}_1 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \mathbf{v}_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
    - singulyar qiymatlar: $\sigma_1 = \sqrt{3}, \sigma_2 = \sqrt{2}$

2. $$\mathbf{\Sigma} = {\displaystyle {\begin{bmatrix} \sqrt{3} & 0 \\ 0 & \sqrt{2} \\ 0 & 0\end{bmatrix}}}$$

3. $$\textbf{V} = {\displaystyle {\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}}}$$

4. $$\begin{split}
\textbf{u}_1 = \frac{1}{\sigma_1} A\textbf{v}_1 = \frac{1}{\sqrt{3}} \left[ \begin{array}{r} 1 \\ -1 \\ 1 \end{array} \right]
\hspace{5mm}
\textbf{u}_2 = \frac{1}{\sigma_2} A\textbf{v}_2 = \frac{1}{\sqrt{2}} \left[ \begin{array}{r} 1 \\ 1 \\ 0 \end{array} \right]
\end{split}$$

$\mathbf{u}_3$ ni aniqlash uchun biz, quyidagi shartlarni bajarish kerak:
- $\mathbf{u}_1$ va $\mathbf{u}_2$ ga ortogonal bo'lishi kerak
- $\{\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3 \}$ vektor to'plamlari $\mathbb{R}^3$ ga bazis ortonolmal bo'lishi kerak. 

Buni amalga oshirish uchun biz $\mathbf{u}_1 \cdot \mathbf{u}_3 = 0$ va $\mathbf{u}_2 \cdot \mathbf{u}_3 = 0$ tenglamalardan tashkil topgan chiziqli tenglamalar sistemasini tuzamiz:
$$
\begin{split}
\left[ \begin{array}{rrr|r} 1 & -1 & \phantom{+}1 & 0 \\ 1 & 1 & 0 & 0 \end{array} \right]
\hspace{5mm}
\Rightarrow
\hspace{5mm}
\textbf{u}_3 = \frac{1}{\sqrt{6}} \left[ \begin{array}{r} -1 \\ 1 \\ 2 \end{array} \right]
\end{split}
$$

Natija:

$$\begin{split}
\mathbf A = \mathbf U \Sigma \mathbf {V}^T
=
\left[ \begin{array}{rcr} 1/\sqrt{3} & 1/\sqrt{2} & -1/\sqrt{6} \\ -1/\sqrt{3} & 1/\sqrt{2} & 1/\sqrt{6} \\ 1/\sqrt{3} & 0 & 2/\sqrt{6} \end{array} \right]
\left[ \begin{array}{rr} \sqrt{3} & 0 \\ 0 & \sqrt{2} \\ 0 & 0 \end{array} \right]  \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}^T
\end{split}$$

```python
import numpy as np

A = np.array([[1, 1], [1, -1], [0, 1]])

U, S, VT = np.linalg.svd(A)

print("U:\n", U)
print("S:\n", S)
print("VT:\n", VT)

```