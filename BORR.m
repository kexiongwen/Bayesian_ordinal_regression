function[beta_sample,b_sample,cutpoints_sample]=BORR[Y,X,T,sparse_T]

M=10000;
burn_in=10000;
v=6.424;
s=1.54;
T1=1e-2;
S1=size(X);
S2=size(T);
d=1;

beta_sample=zeros(S1(2),M+burn_in);
b_sample=ones(S2(2),M+burn_in);
tau_sample=ones(S1(2),1);
v_sample=ones(S1(2),1);
Z_sample=ones(S1(1),1);
omega_sample=ones(S1(1),1);
lam_sample=1;
phi_sample=1;
Lambda_sample=eye(S2(2));
Lambda_cholesky_sample=eye(S2(2));
Precision=speye(S1(2)+S2(2));

e2=ones(S1(2)+S2(2),1);
Mask1=zeros(S1(2),1);

cutpoints_sample=repmat(linspace(-1,1,max(Y)+1)',1,M+burn_in);
cutpoints_sample(1,:)=-Inf;
cutpoints_sample(end,:)=Inf;

for i=2:(M+burn_in)

    %beta_sample
    G=(tau_sample)./lam_sample.^2;
    Mask1=G<T1;

    %Weight
    D=sqrt(omega_sample);

    %Preconditioning feature matrix
    XTD=X'.*D';
    GTTD=T'.*D';
    GXTD=G.*XTD;
    DZ=D.*Z_sample;

    %Preconditioning precision matrix
    Precision(1:P,1:P)=GXTD*GXTD'.*(1-Mask1*Mask1')+speye(S(2));

    Precision(P+1:P+Q,P+1:P+Q)=GTTD*GTTD'+Lambda_sample;
    GTTDXG=GTTD*GXTD';
    Precision(1:P,P+1:P+Q)=GTTDXG';
    Precision(P+1:P+Q,1:P)=GTTDXG;

    %Sample e
    e1=randn(S1(1),1);
    e2(1:S1(2))=GXTD*DZ+GXTD*e1+randn(S1(2),1);
    e2(S1(2)+1:S1(2)+S2(2))=GTTD*DZ+GTTD*e1+Lambda_cholesky_sample*randn(S2(2),1);

    %Solve Preconditioning the linear system by conjugated gradient method
    beta_tilde=cgs(sparse(GXTDXG.*(1-Mask1*Mask1')+speye(S2(2))),b,1e-3);

    %revert to the solution of the original system
    beta_sample(:,i)=G.*beta_tilde;

    % Sampling lambda
    lam_sample=gamrnd(2*S(2)+0.5,1./(sum(sqrt(abs(beta_sample(:,i))))+1./a_sample));

    % Sampling phi
    a_sample=1./gamrnd(1,1./(1+lam_sample));

    temp=lam_sample.^2*abs(beta_sample(:,i));

    % Sampling V
    v_sample=2./random('InverseGaussian',1./sqrt(temp),1);
    
    % Sampling tau
    tau_sample=v_sample./sqrt(random('InverseGaussian',v_sample./temp,1));

    % Sample Z
    Mean=X*beta_sample(:,i)+T*b_sample(:,i);

    low1=()
    




























end













end