(Diff1) $p_\theta\left(\mathbf{x}_0\right)=\int p_\theta\left(\mathbf{x}_{0 : T}\right) d \mathbf{x}_{1: T}$.

(VAE1) $p(\mathbf{x})=\int p(\mathbf{x}, \mathbf{z}) d \mathbf{z}$



(Diff2) $\log p_\theta\left(\mathrm{x}_0\right)=\log \int p_\theta\left(\mathrm{x}_{0 : T}\right) d \mathrm{x}_{1: T}$.

(VAE2) $\log p(\mathbf{x})=\log \int p(\mathbf{x}, \mathbf{z}) d \mathbf{z}$.



(Diff3) $\log p_\theta\left(\mathbf{x}_0\right)=\log \int q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right) \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}  d \mathbf{x}_{1: T}$.

(VAE3) $\log p(\mathbf{x})=\log \int q(\mathbf{z} \mid \mathbf{x}) \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x})} d \mathbf{z}$.



( Jensen: $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ )

(Diff4) $\log p_\theta\left(\mathrm{x}_0\right) =\log \int q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right) \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}  d \mathbf{x}_{1: T} \geq  \int q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right) \log \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}  d \mathbf{x}_{1: T}$ 

(VAE4) $
\log p(\mathbf{x})=\log \int q(\mathbf{z} \mid \mathbf{x}) \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x})} d \mathbf{z} 
\geq \int q(\mathbf{z} \mid \mathbf{x}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z} \mid \mathbf{x})} d \mathbf{z}$.



(Diff5) $\mathcal{L}(\mathbf{x})=\mathbb{E}_q\left[-\log \frac{p_\theta\left(\mathbf{x}_{0: T}\right)}{q\left(\mathbf{x}_{1: T} \mid \mathbf{x}_0\right)}\right]$.

(VAE5) $\mathcal{L}(\mathbf{x})=\mathbb{E}_{q(\mathbf{z} \mid \mathbf{x})}[-\frac{\log p(\mathbf{x}, \mathbf{z})}{\log q(\mathbf{z} \mid \mathbf{x})}]$.





