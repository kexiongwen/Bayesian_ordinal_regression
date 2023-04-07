import numpy as np
from scipy import sparse
from scipy.linalg import block_diag
from scipy.sparse.linalg import cg
from scipy.sparse import spdiags
from scipy.stats import logistic
from scipy.stats import invgamma
from scipy.stats import invgauss
from scipy.stats import beta as beta_d

def trunc(a,low,upp):
    
    if (upp-low)>0.4 or a==1:
        
        x=beta_d.ppf((beta_d.cdf(upp,a,a)-beta_d.cdf(low,a,a))*np.random.rand()+beta_d.cdf(low,a,a),a,a)
        
    else:
        
        ink1=(upp**a-low**a)
        ink2=(1-low)**(a-1)
        ink3=(1-upp)**(a-1)
        ink4=ink2-ink3
        ink5=ink4/ink1
        cut=ink1*ink3
        C=0.5*ink1*(ink2+ink3)

        u1=np.random.rand()

        I=0
        while I==0:

            if u1<(cut/C):
                proposal=C*u1/ink1
                ratio=1
            else:
                C1=0.5*ink3**2+ink5*C*u1
                proposal=ink2-(ink2**2-2*C1)**0.5
                ratio=ink5*((1-proposal**(1/(a-1)))**a-low**a)/(ink2-proposal)

            if ratio<np.random.rand():
                continue
            else:
                I=1
                u=proposal

        z=min(upp,1-u**(1/(a-1)))
        x=((np.random.rand()*(z**a-low**a))+low**a)**(1/a)

    return x


def Bayesian_Ordinal_CLM_PO(Y,X,T,M=10000,burn_in=5000,alpha=1):
    
    v=6.424
    s=1.54
    
    N,P=np.shape(X)
    _,Q=np.shape(T)
    d=1
    F=np.concatenate((X, T), axis=1)
 
    #Initialization
    beta_sample=np.ones((P,M+burn_in))
    b_sample=np.ones((Q,M+burn_in))
    beta_b=np.ones(P+Q)
    tau_sample=np.ones(P)
    v_sample=np.ones(P)
    Z_sample=np.zeros((N,1))
    omega_sample=np.ones(N)
    lam_sample=1
    phi_sample=1
    beta_sample[:,0:1]=np.random.randn(P,1)
    Lambda_sample=np.diag(np.ones(Q))
    Lambda_cholesky_sample=np.diag(np.ones(Q))
    
    cutpoints_sample=np.linspace(-1*np.ones(M+burn_in), np.ones(M+burn_in), num=(np.unique(Y).size+1))
    cutpoints_sample[0,:]=-np.Inf
    cutpoints_sample[-1,:]=np.Inf
      
    #MCMC loop
    
    for i in range(1,M+burn_in):
        
        #Sample beta and b

        #Prior preconditioning matrix from global-local shrinkage
        G_diag=tau_sample/lam_sample**2
        G=sparse.csr_matrix(block_diag(np.diag(G_diag), np.diag(np.ones(Q))))

        #Weight
        D=spdiags((np.sqrt(omega_sample)).ravel(),0,N,N)
      
        #Preconditioning feature matrix
        FTD=sparse.csr_matrix.dot(F.T,D)
        GFTD=sparse.csr_matrix.dot(G,FTD)
        DZ=sparse.csr_matrix.dot(D,Z_sample)
  
        #Preconditioning covariance matrix
        GFTDFG=GFTD@GFTD.T

        #Sample e
        e=GFTD@DZ+GFTD@np.random.randn(N,1)+sparse.csr_matrix.dot(sparse.csr_matrix(block_diag(np.diag(np.ones(P)), Lambda_cholesky_sample)),np.random.randn(P+Q,1))

        #Solve Preconditioning the linear system by conjugated gradient method
        beta_b,_=cg(GFTDFG+sparse.csr_matrix(block_diag(np.diag(np.ones(P)), Lambda_sample)),e.ravel(),x0=beta_b,tol=1e-3)
        
        #revert to the solution of the original system
        beta_sample[:,i]=G_diag*beta_b[0:P]
        b_sample[:,i]=beta_b[P:]
      
        #Sample lambda
        lam_sample=np.random.gamma(2*P+0.5,((np.abs(beta_sample[:,i])**0.5).sum()+1/phi_sample)**-1)
        
        #sample_phi
        phi_sample=invgamma.rvs(1)*(1+lam_sample)
        
        #Sample V
        v_sample=2/invgauss.rvs((lam_sample*np.abs(beta_sample[:,i])**0.5)**-1)
        
        #Sample tau2
        tau_sample=v_sample/np.sqrt(invgauss.rvs(v_sample/(lam_sample**2*abs(beta_sample[:,i]))))
        
        #Sample Z
        Mean=X@beta_sample[:,i]+T@b_sample[:,i]
        
        low1=(cutpoints_sample[:,i-1][Y.astype(int)-1]).ravel()
        upp1=(cutpoints_sample[:,i-1][Y.astype(int)]).ravel()
        Z_sample=(Mean-np.log(1/((logistic.cdf(upp1,loc=Mean,scale=1)-logistic.cdf(low1,loc=Mean,scale=1))*np.random.rand(N)+logistic.cdf(low1,loc=Mean,scale=1))-1)).reshape(N,1)
       
        #Sample w
        omega_sample=np.random.gamma(v*np.ones(N)/2,2/(v*s**2+(Z_sample-Mean)**2))
        
        #Sample Lambda
        A2=(spdiags(np.ones(Q),0,Q,Q)-b_sample[:,i:i+1]@b_sample[:,i:i+1].T/(d+b_sample[:,i:i+1].T@b_sample[:,i:i+1]))/d
        L=np.zeros((Q,Q))
        L[np.triu_indices(Q,1)]=np.random.randn(int(Q*(Q-1)/2))
        L=sparse.csr_matrix(L.T+spdiags(np.random.chisquare(np.arange(Q)+1),0,Q,Q)) 
        A =np.linalg.cholesky(A2)
        Lambda_cholesky_sample=sparse.csr_matrix.dot(A,L)
        Lambda_sample = Lambda_cholesky_sample@Lambda_cholesky_sample.T
       
        #Sample cutpoints
        ink=logistic.cdf(cutpoints_sample[:,i-1])
        
        for j in range(1,np.unique(Y).size):
            
            scale=ink[j+1]-ink[j-1]    
            low2=(logistic.cdf(np.amax(Z_sample[Y.ravel()==j]))-ink[j-1])/scale
            upp2=(logistic.cdf(np.amin(Z_sample[Y.ravel()==(j+1)]))-ink[j-1])/scale
            ink[j]=trunc(alpha,low2,upp2)*scale+ink[j-1]
                  
        cutpoints_sample[:,i]=logistic.ppf(ink)
                   
    #End of MCMC chain    
    
    MCMC_chain=(beta_sample[:,burn_in:],b_sample[:,burn_in:],cutpoints_sample[:,burn_in:])
    
    return MCMC_chain