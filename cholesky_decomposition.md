# Exemple chiffré : Décomposition de Cholesky dans la log-vraisemblance (Claude AI)

## Contexte : cas ultra-simple (n=2)

On pose directement une matrice de covariance $\Sigma_y$ et un vecteur d'observations $y$ :

$$
\Sigma_y = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}, 
\quad 
y = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

---

## Étape 1 : Décomposition de Cholesky

On cherche $L$ triangulaire inférieure telle que $\Sigma_y = LL^T$ :

$$
L = \begin{pmatrix} \ell_{11} & 0 \\ \ell_{21} & \ell_{22} \end{pmatrix}
$$

**Calcul des coefficients :**

En imposant $LL^T = \Sigma_y$, on obtient :

- $\ell_{11}^2 = 2 \quad \Rightarrow \quad \ell_{11} = \sqrt{2}$

- $\ell_{21} \cdot \ell_{11} = 1 \quad \Rightarrow \quad \ell_{21} = \frac{1}{\sqrt{2}}$

- $\ell_{21}^2 + \ell_{22}^2 = 2 \quad \Rightarrow \quad \ell_{22} = \sqrt{2 - \frac{1}{2}} = \sqrt{\frac{3}{2}}$

**Résultat :**

$$
L = \begin{pmatrix} \sqrt{2} & 0 \\ \frac{1}{\sqrt{2}} & \sqrt{\frac{3}{2}} \end{pmatrix}
$$

---

## Étape 2 : Terme quadratique $y^T \Sigma_y^{-1} y$

Au lieu d'inverser $\Sigma_y$, on résout le système triangulaire $Lz = y$ :

$$
\begin{pmatrix} \sqrt{2} & 0 \\ \frac{1}{\sqrt{2}} & \sqrt{\frac{3}{2}} \end{pmatrix}
\begin{pmatrix} z_1 \\ z_2 \end{pmatrix}
= \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

**Solution :**

- $z_1 = \frac{1}{\sqrt{2}}$

- $z_2 = \frac{2 - \frac{1}{\sqrt{2}} z_1}{\sqrt{3/2}} = \frac{2 - \frac{1}{2}}{\sqrt{3/2}} = \frac{3/2}{\sqrt{3/2}} = \sqrt{\frac{3}{2}}$

**Forme quadratique :**

$$
y^T \Sigma_y^{-1} y = z^T z = \left(\frac{1}{\sqrt{2}}\right)^2 + \left(\sqrt{\frac{3}{2}}\right)^2 = \frac{1}{2} + \frac{3}{2} = 2
$$


## 🔑 Pourquoi $y^T \Sigma_y^{-1} y = z^T z$ ?

**Démonstration :**

On sait que $\Sigma_y = LL^T$. Donc :

$$
\Sigma_y^{-1} = (LL^T)^{-1} = (L^T)^{-1} L^{-1}
$$

On définit $z$ par la résolution de $Lz = y$, c'est-à-dire :

$$
z = L^{-1} y
$$

> "$z$ est le vecteur qui vérifie $Lz = y$"

Maintenant, calculons la forme quadratique :

$$
\begin{align}
y^T \Sigma_y^{-1} y &= y^T (L^T)^{-1} L^{-1} y \\
&= y^T (L^T)^{-1} \underbrace{L^{-1} y}_{= z} \\
&= \underbrace{(L^{-1} y)^T}_{= z^T} z \\
&= z^T z
\end{align}
$$

**Remarque importante** : $(L^{-1} y)^T = y^T (L^{-1})^T = y^T (L^T)^{-1}$

---

## Étape 3 : Calcul du déterminant

Le déterminant de $\Sigma_y$ s'obtient via les éléments diagonaux de $L$ :

$$
\log|\Sigma_y| = 2 \log(\ell_{11}) + 2 \log(\ell_{22})
$$

$$
= 2\log(\sqrt{2}) + 2\log\left(\sqrt{\frac{3}{2}}\right) = \log(2) + \log\left(\frac{3}{2}\right) = \log(3)
$$

Donc $|\Sigma_y| = 3$.

---

## Log-vraisemblance finale

La log-vraisemblance gaussienne s'écrit :

$$
\log p(y) = -\frac{1}{2} \Big( n\log(2\pi) + \log|\Sigma_y| + y^T\Sigma_y^{-1} y \Big)
$$

Avec $n=2$ :

$$
\log p(y) = -\frac{1}{2} \Big( 2\log(2\pi) + \log(3) + 2 \Big)
$$

---

## 💡 Résumé intuitif

**Pourquoi utiliser Cholesky ?**

✅ **Stabilité numérique** : pas d'inversion explicite de $\Sigma_y$  
✅ **Efficacité** : calcule simultanément $\log|\Sigma_y|$ et $y^T\Sigma_y^{-1} y$  
✅ **Une seule décomposition** : $L$ sert pour les deux termes critiques

**Prochaine étape possible :** exemple complet avec matrice de design $X$ et structure $\Sigma_y = \sigma^2 I + \tau^2 XX^T$.