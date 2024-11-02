# Adversarial Attack

Adversarial attack vs defense

- Adversarial attack

- Adversarial defense



Adversarial attack

- Untargeted attack vs. Targeted attack
- White-box attack vs. Black-box attack



Error magnitude

- $d\left(x, x^{\prime}\right)<\varepsilon$.



Fast gradient sign method (FGSM)

- $x^{\prime}=x+\alpha \cdot \operatorname{sign}\left(\nabla_x L(f(x), y)\right)$.



Projected gradient descent (PGD)

- $x^{t+1}=\operatorname{Proj}_{x+S}\left(x^t+\alpha \cdot \operatorname{sign}\left(\nabla_x L L\left(f\left(x^t\right), y\right)\right)\right)$.



Adversarial Defense

- Adversarial Training
- Generative Model
- Large Margin Training
- Certified Defense



# OOD detection

Closed-world vs. Open-world



Out-of-class(semantic)

- Anomaly detection
- Novelty detection
  - One-class novelty ( cat vs. ?? )
  - Multi-class novelty ( cat vs. dog vs. ?? )



Out-of-domain  ... (same class)

- Novel domain detection도 있음



Open set recognition: (1) + (2)

- (1) Classification
- (2) Out-of-class detection

![image-20241029141319186](/Users/seunghan96/Library/Application Support/typora-user-images/image-20241029141319186.png)

<br>

Threshold-based OOD

- Confidence score

- Maximum Softmax Probability (MSP) ( = baseline )

<br>

DNNs are overconfident

- Expected Calibration error (ECE)

- $\mathrm{ECE}=\sum_{m=1}^M \frac{\left|B_m\right|}{n}\left|\operatorname{acc}\left(B_m\right)-\operatorname{conf}\left(B_m\right)\right|$.

<br>

Temperature scaling = post-processing

<br>

ODIN: Enhanced OOD Detector by Post-Processing

<br>

Mahalnobis Detector

