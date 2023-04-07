# Bayesian ordinal regression with random effect 

## PCG Sampler

$$
\begin{aligned}
& \mbox{Sample} \,\, \beta,b \mid W,\tau^{2},Z,\lambda,\Lambda  \sim \mathrm{N_{p+q}}((\tilde{X}^{T}W \tilde{X}+C)^{-1}\tilde{X}^{T}WZ,(\tilde{X}^{T} W\tilde{X}+C)^{-1})\\
& \mbox{Sample} \,\,\Lambda \mid b \sim \mathrm{Wishart}(t+1,(dI_{q}+bb^{T})^{-1})\\
& \mbox{Sample} \,\,\lambda \mid \beta \sim \mathrm{Gamma}(2p+0.5,\sum_{k=1}^{p}|\beta_{k}|^{\frac{1}{2}}+\frac{1}{\phi})\\
& \mbox{Sample} \,\, \phi \mid \lambda \sim \mathrm{InvGamma}(1,1+\lambda)\\
& \mbox{Sample} \,\, \frac{1}{v_{k}} \mid \beta_{k},\lambda  \sim \mathrm{InverseGaussian}\left(\sqrt{\frac{1}{4\lambda^{2}|\beta_{k}|}},\frac{1}{2}\right), \quad k=1,\dots,p\\
& \mbox{Sample} \,\,\frac{1}{\tau_{k}^{2}} \mid \beta_{k},v_{k},\lambda \sim  \mathrm{InverseGaussian}\left(\frac{1}{\lambda^{2}v_{k}|\beta_{k}|},\frac{1}{v_{k}^{2}}\right), \quad k=1,\dots,p\\
& \mbox{Sample} \,\,z_{i} \mid  y_{i},\beta,b,\gamma  \sim \mathrm{Logistic}(x_{i}\beta+T_{i}b,1)1_{\gamma_{s-1}<z_{i}<\gamma_{s}}, \,\,\, \text{if} \,\,\,y_{i}=s, \,\,\,\text{for} \,\,\,  i=1,\dots,n\\
& \mbox{Sample} \,\,w_{i} \mid z_{i},\beta,b \sim \mathrm{Gamma}\left(\frac{\nu}{2},\frac{\nu\eta^{2}+(z_{i}-x_{i}\beta-T_{i}b)^{2}}{2}\right), \quad i=1,\dots,n\\& \mbox{Sample} \,\, \gamma_{s} \sim \pi(\gamma_{s}|\gamma_{-s},Y,Z) \propto 
f(\gamma_{s})1_{l_{s}<\gamma_{s}<u_{s}}
\end{aligned}
$$

where  $\tilde{X}= (X,T)$ and $C=\left(\begin{array}{cc}
\lambda^4 D_{\tau^2}^{-1} & 0 \\
0 & \Lambda
\end{array}\right)$

##  Conjugated gradient, prior precondition and sparse linear system approximation

### Conjugated gradient

The following procedure generates a sample $\beta$ and $b$  from

$$
\beta,b \mid W,\tau^{2},Z,\lambda,\Lambda  \sim \mathrm{N_{p+q}}((\tilde{X}^{T}W \tilde{X}+C)^{-1}\tilde{X}^{T}WZ,(\tilde{X}^{T} W\tilde{X}+C)^{-1})
$$

1. Generate $e \sim \mathcal{N}\left(\tilde{X}^{T} W Z, \Phi\right)$ by sampling independent Gaussian vectors $\eta \sim \mathcal{N}\left(0, I_{n}\right)$  and $\delta \sim \mathcal{N}\left(0, I_{p+q}\right)$

   

$$
e=\tilde{X}^{T}WZ+\tilde{X}^{T} W^{1 / 2} \eta+\left(\begin{array}{cc}
\lambda^2 D_{\tau^2}^{-1 / 2} & 0 \\
0 & \Lambda^{1 / 2}
\end{array}\right) \odot \delta
$$



where $\Phi=\tilde{X}^T W \tilde{X}+\left(\begin{array}{cc}
\lambda^2 D_{\tau^2} & 0 \\
0 & \Lambda
\end{array}\right)$



2. Solve the following linear system

   

$$
\Phi \theta =e
$$



where $\theta^{T}=(\beta^{T},b^{T})$.  Since $\Phi$ is symmetric and positive-definite, solving the linear system above can be speed up by using the conjugated gradient method.


 ### Prior preconditioning 

Now we consider to use the global and local shrinkage parameters to precondition the linear system $\Phi \theta =e$ to accelerate the  convergence of conjugated gradient. In high-dimensional and very sparse setting,  the covariance matrix $(\boldsymbol{\tilde{X}}^{T} \boldsymbol{W}\boldsymbol{\tilde{X}}+C)^{-1}$ will near to singular. The prior preconditioning approach can also improve the numerical stable of the PCG sampler.

A preconditioner is a positive definite matrix $M$ chosen so that the preconditioned system

$$
\tilde{\Phi} \tilde{\theta}=\tilde{e} \quad \text{for} \quad \tilde{\Phi}=M^{-1 / 2}\Phi M^{-1 / 2} \quad \text{and} \quad \tilde{e}=M^{-1 / 2} e
$$

where $M=\left(\begin{array}{cc}
\lambda^4 D_{\tau^2}^{-1} & 0 \\
0 & I_q
\end{array}\right)$. By setting $\theta=M^{-1/2}\tilde{\theta}$,  we obatin the solution of the original linear system.	

The prior-preconditioned matrix is given by

$$
\tilde{\Phi}=M^{-1/2}\tilde{X}^{T}W\tilde{X}M^{-1/2}+\left(\begin{array}{cc}
I_p & 0 \\
0 & \Lambda
\end{array}\right)
$$

The prior-preconditioned vector is given by

$$
\tilde{e}=\left(\begin{array}{cc}
\lambda^{-2} D_\tau & 0 \\
0 & I_q
\end{array}\right) \tilde{X}^{T}WZ+\left(\begin{array}{cc}
\lambda^{-2} D_\tau & 0 \\
0 & I_q
\end{array}\right) \tilde{X}^{T} W^{1 / 2} \eta+ \left(\begin{array}{cc}
I_p & 0 \\
0 & \Lambda^ {1/2}
\end{array}\right)\delta
$$

### Sparse linear system approximation

By using a user-deﬁned thresholding parameter $\Delta$, we can have sparse approximation for $\tilde{\Sigma}^{-1}$, such that


$$
\begin{aligned}
{\tilde{\Sigma}^{-1}_{\Delta}}_{ij}= &
\begin{cases} 
\left(\lambda^{-2} \tau_i\right)\left(\lambda^{-2} \tau_{j}\right)\left(X^{T}D X\right)_{i j} & \text { if }  \lambda^{-2} \tau_i>\Delta \quad \text{or}\quad \lambda^{-2} \tau_{j}>\Delta\\ 
0 & \text { else } 
\end{cases}\\
{\tilde{\Sigma}^{-1}_{\Delta}}_{ii}= &
\begin{cases} 
\left(\lambda^{-4} \tau_{i}^{2}\right)\left(X^{T}D X\right)_{i i}+1 & \quad\quad \text { if }  \lambda^{-2} \tau_i>\Delta\\
1 & \quad\quad\text { else } 
\end{cases}
\end{aligned}
$$

Therefore, we obtain a three-step procedure to sample the condition posterior of $(\beta,b)$.

1. Generate $\tilde{e}_{\Delta} \sim \mathcal{N}\left(\begin{pmatrix} \lambda^{-2}D_{\tau}^{-1} & 0 \\  0& I_{q}  \end{pmatrix}\tilde{X}^{T} W Z, \tilde{\Phi}\right)$ by using equation $(\ref{eq:e_tilde})$.

2. Use conjugated gradient method to solve the following linear system for $\bar{\theta}_{\Delta}$:

$$
\tilde{\Phi}_{\Delta}\tilde{\theta}_{\Delta}=\tilde{e}
$$

3. Setting $\theta_{\Delta}=\begin{pmatrix} \lambda^{-2}D_{\tau}^{-1} & 0 \\  0& I_{q}  \end{pmatrix}\tilde{\theta}_{\Delta}$, then we have

   

$$
\theta_{\Delta} \sim \mathcal{N}\left(\left(\begin{array}{cc}
\lambda^{-2} D_\tau^{-1} & 0 \\
0 & I_q
\end{array}\right)\tilde{\Phi}_{\Delta}^{-1} X^{T} W Z, \left(\begin{array}{cc}
\lambda^{-2} D_\tau^{-1} & 0 \\
0 & I_q
\end{array}\right)\tilde{\Phi}_{\Delta}^{-1}\tilde{\Phi}\tilde{\Phi}_{\Delta}^{-1}\left(\begin{array}{cc}
\lambda^{-2} D_\tau^{-1} & 0 \\
0 & I_q
\end{array}\right)\right)
$$

### Sampling $\Lambda^{1/2}$ and $\Lambda$

Now we discuss how to sample $\Lambda^{1/2}$ and $\Lambda$ from $\boldsymbol{\Lambda}|\boldsymbol{b} \sim \mathrm{Wishart}(t+1,(dI_{q}+bb^{T})^{-1})$  efficiently. Note that the Bartlett decomposition of a matrix $\Lambda$ from a $q$-variate Wishart distribution with scale matrix $V$ and $n$ degrees of freedom is the factorization:

$$
\Lambda=LAA^{T}L^{T},
$$

where $L$ is the Cholesky factor of $(dI_{q}+bb^{T})^{-1}=\frac{1}{d}I_{q}-\frac{bb^{T}}{d^{2}+d\|b\|_{2}^{2}}$, and

$$
A=\left(\begin{array}{ccccc}
c_1 & 0 & 0 & \cdots & 0 \\
n_{21} & c_2 & 0 & \cdots & 0 \\
n_{31} & n_{32} & c_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
n_{p 1} & n_{p 2} & n_{p 3} & \cdots & c_p
\end{array}\right)
$$

where  $c_i^2 \sim \chi_{n-i+1}^2 \text { and } n_{i j} \sim N(0,1)$ independently. This provides a useful method for obtaining random samples from a Wishart distribution.



### Extra approximation to improve the numerical stability 

In PCG sampler scheme for $L_{1/2}$ prior, we need to sample 



$\frac{1}{v_{j}} \sim \operatorname{InvGaussian}\left(\sqrt{\frac{1}{4 \lambda^{ 2}\left|\beta_{j}\right|}}, \frac{1}{2}\right) \quad \text{and}  \quad \frac{1}{\tau_{j}} \sim \operatorname{InvGaussian}\left(\frac{1}{\lambda^{2} v_{j}\left|\beta_{j}\right|}, \frac{1}{v_{j}}\right)$



When $\lambda^{2}|\beta_{j}| \rightarrow 0$ ,  the mode, the mean and the variance of conditional posterior of $\frac{1}{v_{j}}$ and $\frac{1}{\tau_{j}^{2}}$ will tend to infinte. In high dimension and very sparse setting, in very rarely case this will lead to the numerical instable problem. Python will report divide by zero encountered in true divide when evaulating $\sqrt{\frac{1}{4\lambda^{2}|\beta_{j}|}}$ or $\frac{1}{{\lambda}^{2}v_{j}|\beta_{j}|}$. 



We see that 

$\pi(v_{j} \mid \beta_{j},\lambda)\propto \pi(\beta \mid \lambda, v_{j})\pi(v_{j}) \propto \exp\left(-\frac{\lambda^{2}|\beta_{j}|}{v_{j}}-\frac{1}{4}v_{j}\right)v_{j}^{-1/2}$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \propto \pi(\beta_{j} \mid \tau_{j}^{2},\lambda)\pi(\tau_{j}^{2}\mid v_{j}) \propto \tau_{j}^{-1}\exp \left(-\frac{\lambda^{4}\beta_{j}^{2}}{\tau_{j}^{2}}-\frac{\tau_{j}^{2}}{2v_{j}^{2}}\right)$



If $\lambda^{2}|\beta_{j}| \rightarrow 0$,  then we have $\pi(v_{j} \mid \beta_{j},\lambda) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ and $\pi(\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j}) \rightarrow \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$. 

Thus to improve numerical stability,  we define another thresholding parameter $\Delta$.  If $\lambda^{2}|\beta_{j}|<\Delta$,  we will 

Sample  $v_{j} \mid \beta_{j},\lambda \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{4})$ 

Sample $\tau_{j}^{2} \mid \lambda,\beta_{j},v_{j} \sim \mathrm{Gamma}(\frac{1}{2},\frac{1}{2v_{j}^{2}})$

In practice, we find that, by setting $\Delta \leq 1e^{-5}$,  the resulting approximation error is negligible.
