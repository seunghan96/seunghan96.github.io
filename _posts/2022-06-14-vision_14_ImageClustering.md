# Image Clustering

## (1) Mahalanobis Distance

**Euclidean** vs **Mahalanobis** Distance

Example )

![figure2](/assets/img/cv/cv224.png)

<br>

Euclidean : $$D_{E}\left(\boldsymbol{x}_{1}, \boldsymbol{x}_{2}\right) < D_{E}\left(\boldsymbol{x}_{3}, \boldsymbol{x}_{4}\right)$$

- $$D_{E}\left(x_{i}, x_{j}\right)=\sqrt{\left(x_{i}-x_{j}\right)^{\top}\left(x_{i}-x_{j}\right)}$$.

<br>

Mahalanobis : $$D_{M}\left(x_{1}, x_{2}\right)>D_{M}\left(x_{3}, x_{4}\right)$$

-  $$D_{M}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\sqrt{\left(\boldsymbol{x}_{i}-\boldsymbol{x}_{\boldsymbol{j}}\right)^{\top} M\left(\boldsymbol{x}_{i}-\boldsymbol{x}_{\boldsymbol{j}}\right)}$$.

<br>

## (2) K-means Clustering

- clustering based on **centroid**
- framework : **EM algorithm**
  - E step : finding centroid
  - M step : assigning data to centroids

$$\begin{gathered}
X=C_{1} \cup C_{2} \ldots \cup C_{K}, \quad C_{i} \cap C_{j}=\phi \\
\operatorname{argmin}_{C} \sum_{i=1}^{K} \sum_{x_{j} \in C_{i}} \mid \mid x_{j}-c_{i} \mid \mid ^{2}
\end{gathered}$$.

<br>

![figure2](/assets/img/cv/cv225.png)

<br>

## (3) Unsupervised Metric Learning

Why not find **embedding space**, using metric learning **w.o labeled data**?

( + previous **supervised metric learning** may cause overfitting! )

<br>

Solution

- step 1) pre-train with **UN-labeled** images
  - contrastive learning, using **hard positives & hard negatives** in manifold
- step 2) fine-tune with **labeled** images

![figure2](/assets/img/cv/cv226.png)

<br>

Example)

- contrastive loss :
  - $$l_{c}\left(\mathbf{z}^{r}, \mathbf{z}^{+}, \mathbf{z}^{-}\right)= \mid \mid \mathbf{z}^{r}-\mathbf{z}^{+} \mid \mid ^{2}+\left[m- \mid \mid \mathbf{z}^{r}-\mathbf{z}^{-} \mid \mid _{+}^{2}\right]$$.
- triplet loss :
  - $$l_{t}\left(\mathbf{z}^{r}, \mathbf{z}^{+}, \mathbf{z}^{-}\right)=\left[m+ \mid \mid \mathbf{z}^{r}-\mathbf{z}^{+} \mid \mid ^{2}- \mid \mid \mathbf{z}^{r}-\mathbf{z}^{-} \mid \mid \right]_{+}^{2}$$.

<br>

### Ground Truth (O)

![figure2](/assets/img/cv/cv227.png)

<br>

### Ground Truth (X)

![figure2](/assets/img/cv/cv228.png)

<br>

Conclusion

- without **discrete category label**, use **fine-grained similarity**
- save annotation cost!
- cons :
  - still need some **pre-trained model**!

<br>

### Self-Taught Metric Learning without Labels ( Kim et al., CVPR 2022 )

use **Pseudo-label**!

$$\rightarrow$$ solution : **self-taught networks** ( by **self-knowledge distillation** )

![figure2](/assets/img/cv/cv229.png)

<br>

## (4) t-SNE

t-SNE

= **t-distributed Stochastic Neighbor Embedding**

( dimension reduction for **visualization of high-dim data** )

![figure2](/assets/img/cv/cv230.png)

<br>

( for more details, refer to https://seunghan96.github.io/ml/stat/t-SNE/ )
