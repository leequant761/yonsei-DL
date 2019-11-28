# Bayesian Optimization

with incorporating prior knowledge about optimal value

## 1. Introduction

## 1.1 Calculus-based techniques

mathematical formula of $f(x)$의 property에 기반한 

Linear programming : linearity of objective function

Convex optimization : convexity of objective function

Gradient based optimization(GD, Adam, Newton) : derivative f objective function

## 1.2 Random search techniques, Grid search

브루트포스 알고리즘 처럼 모든 케이스에 대하여 하나씩 찾아가는 과정



## 1.3 문제점

Caluculus-bsed 테크닉 :

> 우리가 objective function의 mathematical formula를 **모르기 **떄문에 적용 불가능

Random-search techniques:

> 모든 경우에 대해서 계산하기 **힘들다**

그래서 efficient search algorithm이 이러한 expensive black-box optimization에 필요하다.

efficiency는 Global과 Local search의 balance와 관련이 있다.

# 2. Bayesian optimization

Bayesian optimization은 black-box optimization을 푸는 framework이다.

## 2.1 Basic concept : surrogate model

Surrogate model은 mathematical data-driven model이다. 

우리는 목적 함수는 모르지만 이를 흉내내는 함수이다. 

Bayesian optimization은 2가지 단계로 이루어져있다.

1. 주어진 데이터 $D_n = (x_i, f(x_i))_{i=1} ^ n$하에서, surrogate model을 업데이트 해라.
2. surrogate model하에서 acquisition function $u(x)$를 계산하고 select $x_{n+1} = argmax_{x\in \mathcal{X}}u(x|D_n)$

3. Augment data $D_{n+1} = (D_n, (x_{n+1}, f(x_{n+1})))$ and repeat 1

## 2.2 Gaussian processes(surrogate)

Gaussian Process는 mean function $\mu$와 kernel function $\kappa$으로 특정되며 다음과 같이 표현한다.

$$
f(\cdot) \sim GP(\mu(\cdot), \kappa(\cdot, \cdot))
$$

where $[f (x_1), \cdots, f(x_n)] ~ \sim \mathcal{MVN} (\boldsymbol \mu, \mathbf{K})$ with $\mathbf{K}_{ij} = \kappa(x_i, x_j)$ and $\boldsymbol{\mu} = [\mu(x_1) \cdots, \mu(x_N)]$

$\kappa$는 positive definite kernel function이다. 

> unknown function을 $f$를 분포로 보는 점에서 이는 베이지안이다.(maybe)
>
> 서칭 범위를 어느정도 주기 떄문에 베이지안이다(? 좀 더 찾아보겠다.)

Bayesian optimization에서 GP는 objective function의 prior로서 사용된다.

GP의 장점은 GP는 inference까지 제공하는 방법이다.

즉, 주어진 $D_n = (x_i, f(x_i))_{i=1}^n$ 하에서, 우리는 posterior distribution을 구할 수 있다.

그로 인하여, 여러 통계량들을 구할 수 있게 된다.

예측단계를 construction하기 위해서 다음과 같은 joint distribution을 생각해보자.
$$
\left(\begin{array}{c}{\mathbf{f}} \\ {\mathbf{f}_{*}}\end{array}\right) \sim \mathcal{N}\left(\left(\begin{array}{c}{\boldsymbol{\mu}} \\ {\boldsymbol{\mu}_{*}}\end{array}\right),\left(\begin{array}{cc}{\mathbf{K}} & {\mathbf{K}_{*}} \\ {\mathbf{K}_{*}^{T}} & {\mathbf{K}_{* *}}\end{array}\right)\right)
$$
where $\mathbf{K} = \kappa(\mathbf{X}, \mathbf{X})_{N\times N}, \quad \mathbf{K_*}= \kappa(\mathbf{X}, \mathbf{X_*})_{N\times N_*} , \quad \mathbf{K_{**}}  = \kappa(\mathbf{X_*}, \mathbf{X_*})_{N_*\times N_*}$

이를 통해 다음과 같은 conditional distribution을 구할 수 있다.
		$\begin{aligned} p\left(\mathbf{f}_{*} | \mathbf{X}_{*}, \mathbf{X}, \mathbf{f}\right) &=\mathcal{N}\left(\mathbf{f}_{*} | \boldsymbol{\mu}_{*}, \mathbf{\Sigma}_{*}\right) \\ \boldsymbol{\mu}_{*} &=\boldsymbol{\mu}\left(\mathbf{X}_{*}\right)+\mathbf{K}_{*}^{T} \mathbf{K}^{-1}(\mathbf{f}-\boldsymbol{\mu}(\mathbf{X})) \\ \mathbf{\Sigma}_{*} &=\mathbf{K}_{* *}-\mathbf{K}_{*}^{T} \mathbf{K}^{-1} \mathbf{K}_{*} \end{aligned}$

일반적으로  $\mu(x)$를 0으로 두었다면 $\boldsymbol{\mu}=0$이다.

​		$\begin{aligned} p\left(\mathbf{f}_{*} | \mathbf{X}_{*}, \mathbf{X}, \mathbf{f}\right) &=\mathcal{N}\left(\mathbf{f}_{*} | \boldsymbol{\mu}_{*}, \mathbf{\Sigma}_{*}\right) \\ \boldsymbol{\mu}_{*} &=\mathbf{K}_{*}^{T} \mathbf{K}^{-1}\mathbf{f} \\ \mathbf{\Sigma}_{*} &=\mathbf{K}_{* *}-\mathbf{K}_{*}^{T} \mathbf{K}^{-1} \mathbf{K}_{*} \end{aligned}$

## 2.3 Bayesian Optimizer

## 2.3.1 Considerations

목표 : Find $x^* = \text{argmax} _ {x\in \mathcal{X}}f(x)$ (물론 f를 모르니깐 문제임)

> 이를 위해서 balance **global** and **local** search for acquisition function가 중요하다.

1. local

   > select $x^*$ so that $\mu_*(\cdot) = \operatorname{E}[f(\cdot | D_n)]$ is high

2. global

   > select $x^*$ so that $\sigma_* (\cdot)^2 = \operatorname{Var}[f(\cdot | D_n)]$ is high $\because$ it has potential large & relatively isolated point

> GP가 무한대에서 평균이 0으로 죽는 것에 대해서 알아오자. (즉 선택이 안된다는 것을)
>
> 아니면 우리가 range를 정해주기 때문에 걱정이 없을 수도?

### 2.3.2 Select querying point under Gaussian process

다음의 querying point의 선택 기준이 되는 acquisition function $u(x)$

Acquisition function $u(x)$ : 우리는 실제 $f$를 모르므로 global optimization을 위한 valuation

1. Upper confidence bound

   $u(x)= \mu^*(x) + k \sigma^*(x)$ where $k$ is hyper-parameter

2. Expected Improvement

   $u(x) =  \mathbb{E}[\max(f(x) - f(x^+), 0)]$ where $x^+ = \operatorname{argmax}_{x_i \in x_{1:n}} f(x_i)$
  
   $u(x) =(\mu_*(x) - f(x^+) - \xi)\times\Phi(Z) + \sigma(\mathbf{x}) \times\phi(Z) $ where $Z = \frac{\mu_*(x) - f(x^+) - \xi}{\sigma_*(x)}$
  
   $\Phi , \phi$ 는 cdf, pdf of gaussian distribution
   
   $\xi$는 크면 클 수록 $\mu_*(\cdot)$ 보다 $\sigma_*(\cdot)$에 더 weight을 주는 hyper-parameter이다.

**Select** $x_{n+1} = \text{argmax }u(x;D_n) \text{ and evaluate   }f(x_{n+1})$

