# Bayesian ordinal regression with random effect 



## Example dataset


|   Y   | count   |
|---------|----------|
|   1   |   4    | 
|   2   |   5    |
|   3   |   9    |
|   4   |   41   |
|   5   |   90   |
|   6   |  143    |
|  Total  |  292    | 

The response here is the overall rating a student has assigned for a particular staff member teaching a particular course. As students can/do evaluate multiple staff members from the same course or across courses, we thus have included random effects for the students, staff, and course. In total there are 109 students who completed the survey for 18 staff members across 14 courses. 



## Model setting

Considering the linear model


$$
Z_{ijk}=x_{ijk}\beta+U_{i}+V_{j}+W_{k}+\epsilon_{ijk}
$$


where $Z_{ijk}$ is the latent variable from the ith student who evaluates jth staff in kth course. $x_{ijk}$ is the $1\times p$ covarate vector and $\beta$ is the $p \times 1$ coefficient vector. The fixed effect is determined by $x_{ijk}\beta$. $U_{i}$ is the student-specific random effect. $V_{j}$ is the staff-specific random effect. $W_{k}$ is the course-specific random effect.



The correspondence between the observed response $y_{ijk}$ and the latent variable $z_{ijk}$ is defined by the unknown cutoff points:



$$
y_{ijk}=s \quad \text{if} \quad \gamma_{s-1}
$$


where $-\infty=\gamma_{0}<\gamma_{1}<\ldots<\gamma_{5}<\gamma_{6}=\infty$.



If we use the probit model, then the error $\epsilon_{ijk}$ is i.i.d standard gaussian distribution. If we use proportionl logit odds, then the error $\epsilon_{ijk}$ is i.i.d standard logistic distribution.



## Compact representation of the model

Now I rewrite the model above in a more compact way, which is useful for me to derive the MCMC scheme and write down the efficient code. 



$$
Z=X\beta+Tb+\epsilon
$$



where $Z$ is $N \times 1$ vector, $X$ is $N \times p$ covarate matrix corresponding to fixed effects. $T$ is $N \times (109+18+14)$ covarate matrix corresponding to random effects. $\epsilon$ is $N \times 1$ error vector. The $(109+18+14) \times 1$ vector $b$ is the random effects.

In particular, we should note the $T$ is the zero-one matrix to select the random effect.

Let's say the $1 \times (109+18+14)$  vector $t_{100}$ is the 100th row of matrix $T$ which is corresponding to 100th observation.Then only 3 elements from $t_{100}$ are one, others are 0. To be more exact, from 1th to 109th elements, only one of them is 1. From 110th to 127th elements,only one of them is 1. From 128th to 141th elements, only one of them is 1.

In plain English, for 100th observation, $t_{100}$ records which student took the evaluation to which staff in which course.

# Prior setting
To introduce sparsity into the model, we choose exponential power prior for the parameters $\beta$


$$
\pi(\beta_{k} \mid \lambda)=\frac{\lambda^{2}}{4}\exp(-\lambda |\beta_{k}|^{\frac{1}{2}})
$$


with hyper-parameter $\sqrt{\lambda} \sim \mathrm{Cauchy}(0,1)$. The exponential power prior with $\alpha=\frac{1}{2}$ has Normal-Exponential -Gamma mixture representation:

$$
\begin{aligned}
&\beta | \tau_{1}^{2}, \ldots, \tau_{p}^{2}  \sim N_{p}\left(\mathbf{0},  \frac{1}{\lambda^{4}}D_{\tau^{2}}\right), \quad D_{\tau^{2}}=\mathrm{diag}\left(\tau_{1}^{2}, \ldots, \tau_{p}^{2}\right) \\ 
&\tau_{1}^{2}|v_{1}^{2}, \ldots, \tau_{p}^{2}|v_{p}^{2}  \sim  \prod_{k=1}^{p} \mathrm{Exp}(\frac{1}{2v_{k}^{2}}) \\ 
&v_{1} ,    \ldots, v_{p}    \sim \prod_{k=1}^{p} \mathrm{Gamma}(\frac{3}{2},\frac{1}{4}) 
\end{aligned}    
$$

For setting the prior of the cutoff points $\gamma=(\gamma_{1},...,\gamma_{S-1})$, first suppose that there is a normal distribution $Z\sim N(0,\sigma_{0}^{2})$ with CDF function $F(\cdot)$, such that the probability at the interval between each cutoff point is 

$p_{s}=P(\gamma_{s-1} < z <\gamma_{s})=F(\gamma_{s})-F(\gamma_{s-1})$ with $j=1,\ldots,S$.  So $\sum_{s=1}^{S}p_{s}=1$. It follows that:



$$
\begin{aligned}
\gamma_{1} &=F^{-1}\left(p_{1}\right) \\
\gamma_{2} &=F^{-1}\left(p_{1}+p_{2}\right) \\
\cdots & \\
\gamma_{S-1} &=F^{-1}\left(p_{1}+p_{2}+\cdots+p_{S-1}\right)
\end{aligned}
$$



Then we assign a  symmetric Dirichlet prior to $(p_{1},p_{2},...,p_{S})$. That is


$$
\pi\left(p_{1}, \ldots, p_{S} | \alpha\right)=\frac{\Gamma(\alpha S)}{\Gamma(\alpha)^{S}} \prod_{s=1}^{S} p_{s}^{\alpha-1}
$$
By transformation from equation (3), we have 


$$
\pi(\gamma_{1},\gamma_{2},...,\gamma_{S-1} \mid \alpha,v)=\frac{\Gamma(\alpha S)}{\Gamma(\alpha)^{S}} \prod_{s=1}^{S}[F(\gamma_{s})-F(\gamma_{s-1})]^{\alpha-1}\prod_{s=1}^{S-1}f(\gamma_{s})
$$


where $f(\cdot)$ is the density of $F(\cdot)$. Since the random effects $b \sim N(0,\Lambda^{-1})$ with unknown precision matrix $\Lambda^{-1}$,we assign a conjugate prior to $\Lambda$.


$$
\Lambda \sim \mathrm{Wishart}(v,P^{-1})
$$


## PCG Sampler

S1. Sample $\beta,b \mid W,\tau^{2},Z,\lambda,\Lambda  \sim \mathrm{N_{p+q}}((\tilde{X}^{T}W \tilde{X}+C)^{-1}\tilde{X}^{T}WZ,(\tilde{X}^{T} W\tilde{X}+C)^{-1})$

S2. Sample  $\Lambda \mid b \sim \mathrm{Wishart}(t+1,(dI_{q}+bb^{T})^{-1})$

S3. Sample  $\lambda \mid \beta \sim \mathrm{Gamma}(2p+0.5,\sum_{k=1}^{p}|\beta_{k}|^{\frac{1}{2}}+\frac{1}{\phi})$

S4. Sample $\phi \mid \lambda \sim \mathrm{InvGamma}(1,1+\lambda)$

S5. Sample  $\frac{1}{v_{k}} \mid \beta_{k},\lambda  \sim \mathrm{InverseGaussian}\left(\sqrt{\frac{1}{4\lambda^{2}|\beta_{k}|}},\frac{1}{2}\right), \quad k=1,\dots,p$

S6. Sample $\frac{1}{\tau_{k}^{2}} \mid \beta_{k},v_{k},\lambda \sim  \mathrm{InverseGaussian}\left(\frac{1}{\lambda^{2}v_{k}|\beta_{k}|},\frac{1}{v_{k}^{2}}\right), \quad k=1,\dots,p$

S7. Sample $z_{i} \mid  y_{i},\beta,b,\gamma  \sim \mathrm{Logistic}(x_{i}\beta+T_{i}b,1)1_{\gamma_{s-1} < z_{i} < \gamma_{s}}, \quad \text{if} \quad y_{i}=s \quad \text{for} \quad  i=1,\dots,n$

S8. Sample $w_{i} \mid z_{i},\beta,b \sim \mathrm{Gamma}\left(\frac{\nu}{2},\frac{\nu\eta^{2}+(z_{i}-x_{i}\beta-T_{i}b)^{2}}{2}\right), \quad i=1,\dots,n$

S8. Sample $\gamma_{s} \sim \pi(\gamma_{s}|\gamma_{-s},Y,Z) \propto f(\gamma_{s})1_{l_{s} < \gamma_{s} < u_{s}}$

where  $\tilde{X}= (X,T)$ and 

$$
C=\left(\begin{array}{cc}
\lambda^4 D_{\tau^2}^{-1} & 0 \\
0 & \Lambda
\end{array}\right)
$$

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



where 

$$
\Phi=\tilde{X}^T W \tilde{X}+\left(\begin{array}{cc}
\lambda^2 D_{\tau^2} & 0 \\
0 & \Lambda
\end{array}\right)
$$



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

where 

$$
M=\left(\begin{array}{cc}
\lambda^4 D_{\tau^2}^{-1} & 0 \\
0 & I_q
\end{array}\right)
$$


By setting $\theta=M^{-1/2}\tilde{\theta}$,  we obatin the solution of the original linear system.	

The prior-preconditioned matrix is given by

$$
\tilde{\Phi}=M^{-1/2}\tilde{X}^{T}W\tilde{X}M^{-1/2}+\left(\begin{array}{cc}
I_p & 0 \\
0 & \Lambda
\end{array}\right)
$$

The prior-preconditioned vector is given by

$$
\tilde{e}=M^{-1/2} \tilde{X}^{T}WZ+M^{-1/2} \tilde{X}^{T} W^{1 / 2} \eta+ \left(\begin{array}{cc}
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



1. Generate 

$$
\tilde{e}_{\Delta} \sim \mathcal{N}\left(M^{-1/2} \tilde{X}^T W Z, \tilde{\Phi}\right)
$$




2. Use conjugated gradient method to solve the following linear system for  $\bar{\theta}_{\Delta}$:

   
$$
\tilde{\Phi}_{\Delta}\tilde{\theta}_{\Delta}=\tilde{e}
$$




3. Setting $\theta_{\Delta}=M^{-1/2}\tilde{\theta}_{\Delta}$ then we have 

   
$$
\theta_{\Delta} \sim \mathcal{N}\left(M^{-1/2}\tilde{\Phi}_{\Delta}^{-1} X^{T} W Z, M^{-1/2}\tilde{\Phi}_{\Delta}^{-1}\tilde{\Phi}\tilde{\Phi}_{\Delta}^{-1}M^{-1/2}\right)
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
