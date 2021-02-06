---
layout: post
title: ColumbiaX - ML - weeks 1 and 2
date: 2018-10-10
use_math: true
image: "bivariategauss2.jpeg"
comments: true
tags: [ols, mle, ridge regression, map]
---
We kick start the course by establishing a link between ordinary least squares (OLS) and maximimum likelihood estimation (MLE) for linear regression, before introducing ridge regression and showing its probabilistic interpretation. Along the way we discuss model complexity via the bias-variance tradeoff.

<!--more-->
<hr class="with-margin">
<blockquote class="tip">
<strong>Using these notes:</strong> for a course overview and guidance on these notes click <a class="reference external" href="/../project_edx_ml/2019/05/16/columbiaX-ML-course-summary">here</a>.
<br>
<br>
<strong>Errata:</strong> all errors are mine and if you do spot anything, however small, please let me know.
</blockquote>

<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Overview</h4>

The big idea from the first two weeks was to show how models such as linear and ridge regression can be viewed from a probabilistic perspective.

We [start](#ols_mle) by showing the connection between least squares (OLS) and maximum likelihood estimation (MLE) for linear regression, proving that the optimal set of weights for both is equal under certain assumptions about the error residuals.

We [then](#rr_map) extend least squares regression to add a regularization term for the weights to arrive at ridge regression (RR). We show how the solution to RR is equal to placing a Gaussian prior over the model's weights and calculating the MAP solution from a Bayesian perspective.

Further [analysis](#analysis_ols_rr) of the least squares and RR solutions reveal a tradeoff between bias and variance and the relationship between the two is then [derived](#bv_tradeoff) for linear regression.

[Mathematical results](#math_details_sec) that are not crucial to the main thrust of the post are pushed into a separate section to keep the flow of the post and more lengthy derivations are given in the [appendix](#appendix).

Key notation and a brief explanation of some of the terminology used in the post can be found in the section [below](#notation).

<a name="ols_mle"></a>
<hr class="with-margin">
<h4 class="header" id="ols_mle">Connecting OLS and MLE</h4>

<blockquote class="tip">
<strong>Summary:</strong> the line of attack is to introduce linear regression and derive its objective term under a least squares assumption before showing this is equivalent to performing maximum likelihood estimation under a Gaussian noise assumption.
<br>
<br>
By equivalent we mean both solutions share the same set of optimal weights: $\mathbf{w}_{LS} = \mathbf{w}_{ML}$.
</blockquote>

##### Motivation

The idea is that least squares has an insightful probabilistic interpretation that allows us to analyze its properties. This probabilistic interpretation is revealed by the MLE approach where we pick a model we think as reasonable for the linear regression problem and ask: what assumptions are we making?

##### Linear Regression

For linear regression we can state problem formulation as

$$
y_{i}=w_{0}+\sum_{j=1}^{d} x_{i j} w_{j}+\epsilon_{i}  \tag{0}
$$

where $\epsilon \sim \mathcal{N}(0, \sigma_i^2)$ is an additive independent identically distributed Gaussian noise.

It is possible and more compact to restate this in matrix form by absorbing the intercept, $w_0$ into the vector $\mathbf{w}$ and adding a column of 1s to $X$ to write as:

$$
\mathbf{y}= X \mathbf{w} + \mathbf{\epsilon}
$$

where $X$ is now $n \times (d+1)$ and $\mathbf{w}$ is a vector of length $(d+1)$.

The prediction we make from linear regression is given by:

$$
\mathbf{\tilde{y}}= X \mathbf{w}
$$

and is a vector of length $n$.

##### OLS objective term

In order to assess how good the predictions for the model are we define a loss function we wish to minimize, $\mathcal{L}$. For ordinary least squares (OLS) the loss can be defined as:

<div class="math">
\begin{align*}
\mathcal{L} &= \sum_{i=1}^{n}\left(y_{i}-\mathbf{x_{i}}^{T} \mathbf{w} \right)^{2} \\[5pt]
&= \|\mathbf{y}-X \mathbf{w} \|^{2} \\[5pt]
&= (\mathbf{y}-X \mathbf{w})^{T}(\mathbf{y}-X \mathbf{w}) \tag{1}
\end{align*}
</div>

depending on notational preferences. So far all we have assumed is a least squares loss function and the goal is to minimize this error for the linear model of choice. In many cases modelling scenarios we might think of this as a reasonable thing to do where we have made no (explicit) assumptions about the data.

Our goal is to return a set of weights from the optimization that are optimal for the above model.

##### OLS optimal weights, $\mathbf{w}_{LS}$

In order to derive the optimal set of weights, $\mathbf{w}_{LS}$, that minimizes the above function $\mathcal{L}$ we take the derivative of $(1)$, set it to 0 and solve for $\mathbf{w}$.

It can be helpful when following the below to always think about whether the dimensions tie up and remember that once we differentiate we will return a vector (as we are simultaneously differentiating w.r.t each element of $\mathbf{w}$), of dimension $(d+1) \times 1$. In order to derive the below we apply some standard [results](#matrix_cook) from matrix calculus.

<div class="math">
\begin{alignat*}{2}

\nabla_{\mathbf{w}} \mathcal{L} &= \nabla_{\mathbf{w}} (\mathbf{y}-X \mathbf{w})^{T}(\mathbf{y}-X \mathbf{w}) \\[5pt]
&= \nabla_{\mathbf{w}} [\mathbf{y}^T\mathbf{y} - \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y} +  \mathbf{w}^T X^T X \mathbf{w}] \\[5pt]
&= 2 X^{T} X \mathbf{w}-2 X^{T} \mathbf{y}=0 \\[5pt]
\Rightarrow \mathbf{w}_{LS} &= \left(X^{T} X\right)^{-1} X^{T} \mathbf{y} \tag{2}

\end{alignat*}
</div>

and so we have obtained an expression for the optimal weights under a least squares assumption. It is worth nothing that this solution assumes $(X^T X)^{-1}$ exists, which [may not always be the case](#invert_X).

##### MLE approach

By contrast the maximum likelihood approach starts by assuming a statistical model (i.e. a generative process) for the data given parameters and under the [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) data assumption we aim to maximize the joint log-likelihood of all the data.

In the case of linear regression the model assumption we make is that the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals) are independent and identically distributed Gaussian noise with variance $\sigma^2$. This allows us to write the joint likelihood of the data, an $n$-dimensional Gaussian, as:

$$
p\left(\mathbf{y} | \boldsymbol{\mu}, \sigma^{2} I \right)=\frac{1}{\left(2 \pi \sigma^{2}\right)^{\frac{n}{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(\mathbf{y}-\boldsymbol{\mu})^{T}\, I \, (\mathbf{y}-\boldsymbol{\mu})\right) \tag{3}
$$

where under the Gaussian noise assumption $\Sigma = \sigma^2 I$ is a diagonal matrix. Recalling that the determinant of a diagonal matrix is the product of its diagonal entries permits the above form and it also pays to note that $(\mathbf{y}-\boldsymbol{\mu})$ has dimensions $n \times 1$ and $\boldsymbol{\mu}$ is the prediction for each observation, also with dimension $n \times 1$. Under the Gaussian noise assumption we are making this is equivalent to setting the prediction, $\boldsymbol{\mu} = X \mathbf{w}$.

There are a few equivalent ways of stating the assumption we make about the data in a MLE setting for linear regression:

<div class="bullet"> 
<ol> 1. $y_i = \mathbf{x_i}^T \mathbf{w} + \epsilon_i$ with $\epsilon_i \stackrel{ind}{\sim} N(0, \sigma^2)$ for $i=1,...n$</ol>
<ol> 2. $y_i \stackrel{ind}{\sim} \mathcal{N}(\mathbf{x_i}^T \mathbf{w}, \sigma^2)$ for $i=1,...n$</ol>
<ol> 3. $\mathbf{y} \sim \mathcal{N}(X \mathbf{w}, \sigma^2 I)$ and so $p(\mathbf{y} \| \mathbf{w}, X) = \mathcal{N}(X \mathbf{w}, \sigma^2 I)$</ol>
</div>
Note this doesn't mean that the target $\mathbf{y}$ itself is normally distributed but that given a mean prediction of $X\mathbf{w}$ the residuals follow a Gaussian distribution assumed to have constant variance.

Having now set up the maximum likelihood model we move on to derive the optimal weights, $\mathbf{w}_{ML}$, for linear regression under MLE with a Gaussian likelihood.

##### MLE optimal weights, $\mathbf{w}_{ML}$

In the [appendix](#mle_solution) we show in general how to derive the MLE solution for a multivariate Gaussian and we follow a similar procedure here and so will omit all the details. The approach is to plug $\boldsymbol{\mu} = X \mathbf{w}$ into $(3)$, takes logs, differentiate w.r.t. $\mathbf{w}$, set to 0 and solve for $\mathbf{w}$. This gives:

<div class="math">
\begin{alignat*}{1}

0 &= \nabla_{\mathbf{w}} \frac{1}{\left(2 \pi \sigma^{2}\right)^{\frac{n}{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(\mathbf{y}-X \mathbf{w})^{T} I(\mathbf{y}-X \mathbf{w})\right) \\[5pt]
&\vdots \\[5pt]
&= -\frac{1}{2 \sigma^{2}} \nabla_{\mathbf{w}} \| \mathbf{y} - X \mathbf{w} \|^{2}
\end{alignat*}
</div>

which immediately we recognise as the same form as $(1)$ up to a constant and so need not calculate any further.

We can thus conclude that $\mathbf{w}\_{LS} = \mathbf{w}\_{ML}$ and see that least squares has a probabilistic interpretation. In this case we reveal that when doing least squares for linear regression we are implicitly assuming that the residuals are independent and identically distributed Gaussian noise with constant variance.

[Analysis](#mean_var_mle_solution) of the variance of the OLS/MLE solution reveal that while the solution is unbiased it can have high variance, with large weight values that are sensitive to $\mathbf{y}$ - ridge regression offers a solution to this shortcoming.

<a name="rr_map"></a>
<hr class="with-margin">
<h4 class="header" id="rr_map">Connecting ridge regression and MAP</h4>

<blockquote class="tip">
<strong>Summary:</strong> ridge regression maximizes the posterior, least squares maximizes the likelihood.
<br>
<br>
We will prove that $\mathbf{w}_{RR} = \mathbf{w}_{MAP}$.
</blockquote>

##### Ridge regression

In general, when developing a model in can be helpful to constrain the model parameters in some way to prevent overfitting. A common technique is to apply a regularization term to the objective function that discourages large values in the weight vector $\mathbf{w}$. For ridge regression this regularization term is added to the least squares objective function as a squared sum of the vector $\mathbf{w}$ with a hyperparameter $\lambda$ controlling the weight this term plays in the ridge regression loss function:

<div class="math">
\begin{align*}
\mathcal{L} &= \|\mathbf{y}-X \mathbf{w}\|^{2}+\lambda\|\mathbf{w}\|^{2}  \tag{4} \\[5pt]
&= (\mathbf{y}-X \mathbf{w})^{T}(\mathbf{y}-X \mathbf{w})+\lambda \mathbf{w}^{T} \mathbf{w}.
\end{align*}
</div>

We can find the optimal weights by the same procedure as for OLS:

$$
\nabla_{\mathbf{w}} \mathcal{L} = -2 X^{T} \mathbf{y}+2 X^{T} X \mathbf{w}+2 \lambda \mathbf{w}=0
$$

leading to the solution for ridge regression:

$$
\mathbf{w}_{RR} = \left(\lambda I+X^{T} X\right)^{-1} X^{T} \mathbf{y}. \tag{5}
$$

As with OLS we didn't explicitly make any probabilistic assumptions, we just defined a loss function with a regularization term and calculated the optimal set of weights for it.

However, as we now show, it turns out the ridge regression solution also has a nice probabilistic interpretation.

##### Ridge regression - Bayesian perspective

We make the connection to a Bayesian perspective by placing a prior over the weights of the solution. In this case we will place a Gaussian prior over the weights with 0 mean:

$$
p(\mathbf{w})=\left(\frac{\lambda}{2 \pi}\right)^{\frac{d}{2}} \mathrm{e}^{-\frac{\lambda}{2} \mathbf{w}^{T} \mathbf{w}}
$$

and so $\mathbf{w} \sim N\left(0, \lambda^{-1}I\right)$.

<blockquote class="tip">
<strong>Sidebar on Bayes' rule for MAP</strong>
<br>
By Bayes' rule the solution that corresponds to the maximum point (the mode) in the posterior distribution is called the <a class="reference external" href="https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation">maximum a posteriori (MAP)</a>
solution.
<br>
<br>
This is analogous to the maximum likelihood solution except we combine the likelihood with a distribution for the model's weights via the prior.
<br>
<br>
Bayes's rule can be written in this case as:

$$
p(\mathbf{w} | \mathbf{y}, X) = \dfrac{p(\mathbf{y} | \mathbf{w}, X) p(\mathbf{w})}{p(\mathbf{y} | X)}
$$

whereby taking logs of both sides leaves us with:

$$
\ln \underbrace{p(\mathbf{w} | \mathbf{y}, X)}_\text{posterior} = \underbrace{\ln p(\mathbf{y} | \mathbf{w}, X)}_\text{log-likelihood} + \ln \underbrace{p(\mathbf{w})}_\text{prior} - \ln \underbrace{ p(\mathbf{y} | X)}_\text{evidence}
$$
<br>
It is worth noting that the MAP solution is still a point estimate and as a result we don't need to calculate the denominator in Bayes' rule, $p(\mathbf{y} | X)$. This is helpful as this term is usually difficult to calculate.
</blockquote>

We proceed by recalling that the likelihood assumption is $p(\mathbf{y} \| \mathbf{w}, X) = \mathcal{N}(X \mathbf{w}, \sigma^2 I)$ which we can solve by taking logs of Bayes' rule.

The MAP solution can thus be written as:

<div class="math">
\begin{align*}
\mathbf{w}_{MAP} &= \arg \max_{\mathbf{w}} \ln p(\mathbf{y} | \mathbf{w}, X)+\ln p(\mathbf{w})  \\[5pt]
&= \arg \max_{\mathbf{w}}-\frac{1}{2 \sigma^{2}}(\mathbf{y}-X \mathbf{w})^{T}(\mathbf{y}-X \mathbf{w})-\frac{\lambda}{2} \mathbf{w}^{T} \mathbf{w}+\mathrm{const.} \\[5pt]
&= \mathcal{L}
\end{align*}
</div>

by plugging in the Gaussian assumptions we have made for the likelihood and prior. We can solve the above analytically as for MLE by differentiating w.r.t $\mathbf{w}$ and setting to 0:

$$
\nabla_{\mathbf{w}} \mathcal{L}=\frac{1}{\sigma^{2}} X^{T} \mathbf{y}-\frac{1}{\sigma^{2}} X^{T} X \mathbf{w}-\lambda \mathbf{w}=0
$$

which results in the MAP solution for ridge regression as:

$$
\mathbf{w}_{MAP} = \left(\lambda \sigma^{2} I+X^{T} X\right)^{-1} X^{T} \mathbf{y}. \tag{6}
$$

This solution is the same as $\mathbf{w}\_{RR}$ (after redefining the constant $\lambda$) and so we have shown that $\mathbf{w}\_{RR} = \mathbf{w}\_{MAP}$.

<blockquote class="tip">
<strong>Note:</strong> when $\lambda = 0$ we have $\mathbf{w}_{OLS} = \mathbf{w}_{RR}$.
</blockquote>

##### Comment

It is possible to extend the above analysis in many ways, in particular it becomes easy to make the link to lasso regression. A similar analysis to the above shows that adding a regularization term that sums the absolute values of $\mathbf{w}$ is equivalent to placing a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution) over the model's weights from a Bayesian perspective.

The regularization term added in the ridge regression objective term is [sometimes called](https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm) a $l_2$ penalty and for lasso it's often called a $l_1$ penalty.

<a name="analysis_ols_rr"></a>
<hr class="with-margin">
<h4 class="header" id="analysis">Analysis of OLS and RR</h4>

##### Introduction

So far in the above we have proceeded somewhat mechanically in order to link OLS to MLE and then RR to MAP without really analysing what we are doing. In this section we provide some analysis of the various solutions to linear regression before proceeding to the bias-variance tradeoff in the next section.

##### Motivating RR

As we have stated the OLS solutions to linear regression can suffer from high variance. In particular, it can be [shown](#mean_var_mle_solution) that the expected value and variance of the MLE solutions to linear regression under the Gaussian assumption, $\mathbf{y} \sim \mathcal{N}\left(X \mathbf{w}, \sigma^{2} I\right)$, are given by:

$$
\mathbb{E}\left[\mathbf{w}_{ML}\right]=\mathbf{w}, \quad \operatorname{Var}\left[\mathbf{w}_{ML}\right]=\sigma^{2}\left(X^{T} X\right)^{-1}.
$$

The above is interpreted as showing that the MLE solution is [unbiased](https://en.wikipedia.org/wiki/Bias_of_an_estimator) but that the model parameters can have large variance if $(X^T X)^{-1}$ is large - typically this is when the columns of $X$ are highly correlated. This can be bad if we want to predict using $\mathbf{w}_{ML}$ as the solution has potentially [very large variance](https://en.wikipedia.org/wiki/Variance_inflation_factor).

Ridge regression adds a penalty term to the objective function of OLS and probabilistically this is equivalent to assuming a Gaussian prior over the model weights. This further assumption about the distribution of the model’s parameters biases the solution we obtain but reduces the variance.

Following similar analysis to ML it can be [shown](#mean_var_rr_solution) that:

$$
\mathbb{E}\left[\mathbf{w}_{RR}\right]=\left(\lambda I+X^{T} X\right)^{-1} X^{T} X \mathbf{w}, \quad \operatorname{Var}\left[\mathbf{w}_{RR}\right]=\sigma^{2} Z\left(X^{T} X\right)^{-1} Z^{T}
$$

where $ Z=\left(I+\lambda\left(X^{T} X\right)^{-1}\right)^{-1}.$

It is noted that $\lambda = 0$ returns the same solutions as in the MLE case. As $\lambda \to \infty$ then $\mathbf{w}\_{RR} \to 0$ and also $\operatorname{Var}\left[\mathbf{w}\_{RR}\right] \to 0 $.

<blockquote class="tip">
<strong>Summary:</strong> the prior assumption we make about the model's parameters biases the solution we obtain as $\mathbb{E}\left[\mathbf{w}_{RR}\right] \neq \mathbf{w}$, when $\lambda \neq 0$.
<br>
<br>
However, the prior assumption reduces the variance of the RR solution and crucially this variance is
<a class="reference external" href="https://www.statlect.com/fundamentals-of-statistics/ridge-regression">always lower</a> than the variance of the MLE solution..
</blockquote>

This trade-off between bias and variance is discussed next.

<a name="bv_tradeoff"></a>
<hr class="with-margin">
<h4 class="header" id="bv">Bias-variance tradeoff</h4>

We now move onto discussing one of the fundamental results in machine learning from a frequentist viewpoint, the bias-variance tradeoff. So far we have talked about the terms bias and variance a little loosely, now we put them on a firmer footing through analysing the generalization error for new data. The ability of a model to generalize to unseen data is of primary importance in machine learning and in order to do this we must try to minimise the overfitting the training data too badly.

The concept of overfitting is linked to model complexity where the loose relationship is that more complex models are more prone to overfitting. The figure below shows a typical case of this phenomena.

<p align="center">
    <img src="/assets/img/bias_variance.png" alt="Image" width="600" height="400" />
</p>
<em class="figure">Bias variance tradeoff</em>

It is worth saying that the problems with maximum likelihood do not arise when we marginalize
over parameters in a Bayesian setting, however this is not discussed in this post and the reader is referred to [PRML](#prml) [3.2].

Recall that in linking OLS to MLE and RR to MAP we have that:
<div class="bullet"> 
<li> Least squares solution: unbiased, but potentially high variance </li>
<li> Ridge regression solution: biased, but lower variance than LS</li>
</div>
<br>
To analyse which of these is preferable we note that ultimately the true thing we care about the generalization error on unseen data. In order to start this analysis we consider the prediction for a single new test prediction: $(\mathbf{x}_0, y_0)$.

<div class="bullet"> 
<li> Least squares predicts: $\mathbf{x}_0^T \mathbf{w}\_{LS}$ </li>
<li> Ridge regression predicts: $\mathbf{x}_0^T \mathbf{w}\_{RR}$ </li>
</div>
<br>
We can calculate the expected squared error of this prediction as:

$$\mathbb{E}[(y_0 - \mathbf{x}_0^T\mathbf{\hat{w}})^2 | X,\mathbf{x}_0] = \int_{\mathbb{R}}^{} \int_{\mathbb{R^n}}^{} (y_0 - \mathbf{x}_0^T\mathbf{\hat{w}})^2 \, \underbrace{p(\mathbf{y} | X, \mathbf{w})}_\text{$ \mathbf{y} \sim \mathcal{N}\left(X \mathbf{w}, \sigma^{2} I\right) $} \, \underbrace{p(y_0 | \mathbf{x}_0, \mathbf{w})}_\text{$ y_0 \sim \mathcal{N}\left(\mathbf{x}_0^T \mathbf{w}, \sigma^{2} \right)$} \, d\mathbf{y} \, dy_0 \tag{7}
$$

where $\mathbf{\hat{w}}$ is either $\mathbf{w}\_{LS}$ or $\mathbf{w}\_{RR}$.

<blockquote class="tip">
<strong>Sidebar on the above integral</strong>
<br>

The above integral looks a little scary and appears to have come out of nowhere. To understand it we first note that it is using the result for expected value:

$$E[g(X) | A] = \int_{} g(x) \, p_{X| A}(x) \, dx$$

which in the notation we have can be written as:

$$E[g(Y_0) | A] = \int_{} g(y_0) \, p_{Y_0 | A}(y_0) \, dy_0.$$

Here we have that:

$$g(y_0) = (y_0 - \mathbf{x}_0^T\mathbf{\hat{w}})^2$$

which is just some function of the random variable $Y_0$. The probability distribution is:

$$
p_{Y_0 | A}(y_0) = \overbrace{\int_{\mathbb{R^n}}^{} \underbrace{p(y_0 | \mathbf{y} , \mathbf{x}_0, \mathbf{w})}_\text{$ = \, p(y_0 | \mathbf{x}_0, \mathbf{w})$} \, p(\mathbf{y} | X, \mathbf{w}) \, d \mathbf{y}}^\text{$ = \, p(y_0 | \mathbf{x}_0, \mathbf{w})$}
$$
and so we could rewrite $(7)$ initially as:

$$
\mathbb{E}[(y_0 - \mathbf{x}_0^T\mathbf{\hat{w}})^2 | X, \mathbf{x}_0] = \int_{\mathbb{R}}^{} (y_0 - \mathbf{x}_0^T\mathbf{\hat{w}})^2 \, p(y_0 | \mathbf{x}_0, \mathbf{w}) \, dy_0
$$

before substituting in the above result.

The above is essentially saying:
<br>
* If we know data $X$ and $\mathbf{x}_0$ and assume there is some true underlying $\mathbf{w}$ (frequentist assumption)
<br>
* Generate $\mathbf{y} \sim \mathcal{N}\left(X \mathbf{w}, \sigma^{2} I\right)$ and approximate $\mathbf{w}$ with $\mathbf{\hat{w}} = \mathbf{w}_{LS}$ or  $\mathbf{\hat{w}} = \mathbf{w}_{RR}$
<br>
* Predict the true target $y_0$ as $\approx \mathbf{x}_0^T \mathbf{\hat{w}}$
<br>
* What is the squared error of the prediction?

</blockquote>

The LHS of $(7)$ can be calculated as:

<div class="math">
\begin{align*}
\mathbb{E}\left[\left(y_{0}-\mathbf{x}_0^{T} \mathbf{\hat{w}}\right)^{2}\right] &= \mathbb{E}\left[y_{0}^{2}\right]-2 \mathbb{E}\left[y_{0}\right] \mathbf{x}_0^{T} \mathbb{E}[\mathbf{\hat{w}}]+\mathbf{x}_0^{T} \mathbb{E}\left[\mathbf{\hat{w}} \mathbf{\hat{w}}^{T}\right] \mathbf{x}_0 \\[5pt]
&= \underbrace{\sigma^{2}}_\text{noise} +
\underbrace{\mathbf{x}_0^{T}(\mathbf{w}-\mathbb{E}[\mathbf{\hat{w}}])(\mathbf{w}-\mathbb{E}[\mathbf{\hat{w}}])^{T} \mathbf{x}_0}_\text{squared bias} + \underbrace{\mathbf{x}_0^{T} \operatorname{Var}[\mathbf{\hat{w}}] \mathbf{x}_0}_\text{variance}

\end{align*}
</div>

where the above is derived with the help of a few results:
<div class="bullet"> 
<li> $\mathbb{E}\left[y_{0} \mathbf{\hat{w}}\right]=\mathbb{E}\left[y_{0}\right] \mathbb{E}[\mathbf{\hat{w}}]$ by independence </li>
<li> $\mathbb{E}\left[y_{0}^{2}\right]=\sigma^{2}+(\mathbf{x}\_0^{T} \mathbf{w})^{2}$ by $y_{0} \sim N\left(\mathbf{x}\_0^{T} \mathbf{w}, \sigma^{2}\right)$ and <a class="reference external" href="{{page.url}}#prob_fact1">prob fact 1</a> </li>
<li> $\mathbb{E}\left[\mathbf{\hat{w}} \mathbf{\hat{w}}^{T}\right]=\operatorname{Var}[\mathbf{\hat{w}}]+\mathbb{E}[\mathbf{\hat{w}}] \mathbb{E}[\mathbf{\hat{w}}]^{T}$ from <a class="reference external" href="{{page.url}}#prob_fact2">prob fact 2</a> </li>
</div>
<br>
##### Comment

We have thus decomposed the prediction error into 3 main components:
<div class="bullet"> 
<ol> 1. Measurement noise – we can’t control this given the model </ol>
<ol> 2. Model bias – how close to the solution we expect to be on average</ol>
<ol> 3. Model variance – how sensitive the solution is to the data</ol>
</div>
<br>
The above analysis is more general (see [ESL](#esl) [7.3]) than the linear regression case though it's usually not possible to get nice equations for the tradeoff.

In the case of OLS and RR we have expressions for $\mathbb{E}[\mathbf{\hat{w}}]$ and $\operatorname{Var}[\mathbf{\hat{w}}]$ and so in theory would be able to compare the generalization error if we knew the true $\mathbf{w}$. In reality we don't and so techniques such as cross-validation are used instead to estimate the generalization error.

When building machine learning predictive models understanding whether you have a bias or a variance problem is critical for deciding what action you should take.

In [week 3](/../project_edx_ml/2018/10/13/columbiaX-ML-week3) we will see how the fully Bayesian treatment of linear regression helps combat overfitting and also offers practical ways to address model complexity. We will also compare lasso and ridge regression and analyse what to do when $d \gg n$.

<a name="math_details_sec"></a>
<hr class="with-margin">
<h4 class="header" id="math_results">Some mathematical results</h4>

<a name="matrix_cook"></a>
##### Matrix cookbook

Some results from [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) are useful.

<a name="mc81"></a>
###### 81

$$
\frac{\partial}{\partial \mathbf{x}} \mathbf{x}^{T} A \mathbf{x} = 2 A \mathbf{x} \tag{MC 81}
$$

when $A$ is symmetric ($A^T = A$).


<a name="mc63"></a>
###### 63
This result is given as:

$$
\frac{\partial \operatorname{trace} \left(A X^{-1} B \right) } {\partial X} = -\left(X^{-1} B A X^{-1}\right)^{T} \tag{MC 63}
$$

and in the use case we have we can assume $A = I = B$ to obtain


$$
\frac{\partial}{\partial \Sigma} \operatorname{trace}(\Sigma^{-1} a) = - \Sigma^{-2} a
$$

where $ a = \sum_{i=1}^{n}\left(\mathbf{x}_{\mathbf{i}}-\boldsymbol{\mu}\right)\left(\mathbf{x}\_{\mathbf{i}}-\boldsymbol{\mu}\right)^{T}$ is of dimension $1 \times 1$ and so can essentially be ignored as a scalar.

<a name="mc57"></a>
###### 57

$$
\frac{\partial}{\partial \Sigma} \ln(|\Sigma|) = \Sigma^{-T} = \Sigma^{-1} \tag{MC 57}
$$

where the last equality holds if $\Sigma$ is symmetric.

<a name="mc69"></a>
###### 69

$$
\frac{\partial \mathbf{x}^{T} \mathbf{a}}{\partial \mathbf{x}}=\frac{\partial \mathbf{a}^{T} \mathbf{x}}{\partial \mathbf{x}}=\mathbf{a} \tag{MC 69}
$$

<a name="trace_trick"></a>
##### Trace trick

The trace of a matrix is the sum of its diagonal entries and so the trace of a scalar is equal to the scalar itself. The trace has the property that:

$$
\operatorname{trace}(ABC) = \operatorname{trace}(CAB) = \operatorname{trace}(BCA)
$$

which is called invariance under cyclical permutations of matrix products. This allows to rewrite the scalar (yes, it's a scalar!):

$$
\left(\mathbf{x}_{\mathbf{i}}-\boldsymbol{\mu}\right)^{T} \Sigma^{-1}\left(\mathbf{x}_{\mathbf{i}}-\boldsymbol{\mu}\right)
$$

as

$$
\operatorname{trace}\left(\Sigma^{-1} \sum_{i=1}^{n}\left(\mathbf{x}_{\mathbf{i}}-\boldsymbol{\mu}\right)\left(\mathbf{x}_{\mathbf{i}}-\boldsymbol{\mu}\right)^{T}\right)
$$

where we then apply $(\text{MC} \, 63)$ [above](#mc63) in order to differentiate.

##### Outer product

Matrix multiplication is the same as summing over all the individual outer products.

$$X^{T}X = \sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^{T}$$

This point will matter when we talk about active learning later in the course.

<a name="prob_facts"></a>
##### Some probability facts

<a name="prob_fact1"></a>
###### Fact 1

The [non-central moment of a Gaussian](https://en.wikipedia.org/wiki/Normal_distribution#Moments) defined by $X \sim \mathcal{N}(\mu, \sigma^2)$ is:

$$
\mathbb{E}[X^2] = \mu^2 + \sigma^2
$$

<a name="prob_fact2"></a>
###### Fact 2

If $y \sim N(\mu, \Sigma)$ then:

<div class="math">
\begin{alignat*}{1}

\operatorname{Var}[y] &= \Sigma  \\[5pt]
&= \mathbb{E}\left[(y-\mathbb{E}[y])(y-\mathbb{E}[y])^{T}\right]  &\text{(by definition)} \\[5pt]
&= \mathbb{E}\left[y y^{T}-y \mu^{T}-\mu y^{T}+\mu \mu^{T}\right]  \hspace{2cm} &\text{(expanding)} \\[5pt]
&= \mathbb{E}\left[y y^{T}\right]-\mu \mu^{T} &\text{(using $\mathbb{E}[y] = \mu$)} \\[5pt]
\Rightarrow \mathbb{E}\left[y y^{T}\right] &= \Sigma+\mu \mu^{T}

\end{alignat*}
</div>

<a name="invert_X"></a>
##### When does $(X^T X)^{-1}$ exist?

When $X^T X$ is full rank. This loosely means that $X$, $n \times (d+1)$, has at least $d+1$ linearly independent rows and so any point in $\mathbb{R}^{d+1}$ can be reached by a weighted combination of $d+1$ rows of $X$.

<a name="notation"></a>
<hr class="with-margin">
<h4 class="header" id="notation">Key notation and terminology</h4>
##### Notation

In general, capital letters are matrices, bold font represents vectors and lower-case letters are scalars. We will also try to introduce new references to notation appropriately to ease reading.
<div class="bullet"> 
<li> $\mathbf{x_i}$: the $i$th observation $\in \mathbb{R}^{d}$ which we think of as a column vector, $d \times 1$ </li>
<li> $X$: an $n \times d$ matrix with each row an observation and each column a different feature, this has $d+1$ columns if we model with an intercept </li>
<li> $\mathbf{y}$: target variable vector, $n \times 1$ </li>
<li> $y_i$: target variable for the $i$th observation, $\in \mathbb{R}$ </li>
<li> $\mathbf{w}$: $d$ dimensional weight vector ($d+1$ if we assume an intercept), $d \times 1$ </li>
</div>
<br>
##### Terminology

###### Maximum-likelihood estimation (MLE)

MLE is a way of estimating the parameters of a model, given data. In some cases this can be done by differentiating the likelihood and solving for the parameter of interest.

$$
\hat{\theta}_{ML} :=\arg \max_{\theta} p\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n} | \theta\right)
$$

###### The notion of a probabilistic model

The broad goal in probabilistic modelling is to choose a family of probability distributions $p(.)$ and learn the parameters of the distribution. A simple example would be learning the mean and covariance of a Gaussian distribution based on data.

###### Bias
Bias is an error from erroneous assumptions in the learning algorithm. High bias models tend to underfit the training data.  

###### Variance
Variance is an error from sensitivity to small fluctuations in the training set. High variance models can fit the noise in a dataset and result in overfitting. Such models perform well on training data but can generalize poorly to unseen data.

<hr class="with-margin">
<h4 class="header" id="references">References</h4>

<div class="bullet"> 
<li>
<a name="prml"></a>
Bishop, C. (2006). Chapters: 1.1, 2.1 - 2.3, 3.1 - 3.2; <a class="reference external" href="https://www.springer.com/gb/book/9780387310732">Pattern Recognition and Machine Learning</a>.</li>
<li>
<a name="esl"></a>
Hastie, T., R. Tibshirani, and J. Friedman (2001). Chapters: 1 - 2, 3.1 - 3.4, 7.1 - 7.3, 7.10;  <a class="reference external" href="http://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a>.</li>
<li>
<a name="edx_ml"></a>
edX, ColumbiaX, <a class="reference external" href="https://www.edx.org/course/machine-learning-1">Machine Learning</a>.</li>
</div>

<hr class="with-margin">
<h4 class="header" id="appendix">Appendix</h4>

<a name="mle_solution"></a>
##### MLE solution for a multivariate Gaussian

<blockquote class="math">
<strong>Approach:</strong> differentiate the joint log-likelihood function to find the MLE parameters.
</blockquote>

Here we derive the maximum likelihood solution under a multivariate Gaussian likelihood assumption for the data.

We start by noting that in general for $n$ [iid](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) data observations $\mathbf{x_i} \in \mathbb{R}^{d}$ the probability of a single observation $p(\mathbf{x} \| \boldsymbol{\mu}, \Sigma)$ can be written as

$$
\mathbf{x_i} \stackrel{\text { iid }}{\sim} p(\mathbf{x} | \boldsymbol{\mu}, \Sigma)
$$

for some probability distribution $p$ with parameters $\theta$. For a multivariate Gaussian $\theta = \\{\boldsymbol{\mu}, \Sigma \\}$.

Under the iid assumption we can write the joint likelihood of all the data as the product of the probability of each individual observation:

$$
p\left(\mathbf{x_1}, \ldots, \mathbf{x_n} | \theta\right)=\prod_{i=1}^{n} p\left(\mathbf{x_i} | \theta\right)
$$

where the goal is the find the optimal values of the parameters $\theta$ that maximize the joint likelihood. That is:

$$
\hat{\theta}_{ML} :=\arg\max_{\theta} p\left(\mathbf{x_1}, \ldots, \mathbf{x_n}| \theta\right).
$$

The joint likelihood is the product of many terms all less than one (they are probabilities) and this can be complicated to compute. We thus take logs to turn multiplication into addition and note this is still maximizing the same set of parameters. This is now the joint log-likelihood and we solve for $\theta$ which amounts to solving:

$$
\sum_{i=1}^{n} \nabla_{(\boldsymbol{\mu}, \Sigma)} \ln p\left(\mathbf{x_i}| \boldsymbol{\mu}, \Sigma\right)=0
$$

###### Solving for the mean vector

<blockquote class="math">
<strong>Summary of math used:</strong> use of log rules, vector matrix multiplication, a <a class="reference external" href="{{page.url}}#matrix_cook">matrix calculus</a> result and noting that $\Sigma$ is positive definite (hence invertible) so we can cancel it.
</blockquote>

<div class="math">
\begin{alignat*}{2}

0 &= \nabla_{\boldsymbol{\mu}} \sum_{i=1}^{n} \ln \frac{1}{\sqrt{(2 \pi)^{d}|\Sigma|}} \exp \left(-\frac{1}{2}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)^{T} \Sigma^{-1}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)\right) &\text{(by definition)} \\[5pt]
&= \nabla_{\boldsymbol{\mu}} \sum_{i=1}^{n}-\frac{1}{2} \ln (2 \pi)^{d}|\Sigma|-\frac{1}{2}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)^{T} \Sigma^{-1}\left(\mathbf{x_i}-\boldsymbol{\mu}\right) &\text{(by log rules)} \\[5pt]
&= -\frac{1}{2} \sum_{i=1}^{n} \nabla_{\boldsymbol{\mu}}\left(\mathbf{x_i}^{T} \Sigma^{-1} \mathbf{x_i}-2 \boldsymbol{\mu}^{T} \Sigma^{-1}\mathbf{x_i}+\boldsymbol{\mu}^{T} \Sigma^{-1} \boldsymbol{\mu}\right) &\text{(expanding brackets)} \\[5pt]
&= -\Sigma^{-1} \sum_{i=1}^{n}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)  &\text{(diff. w.r.t $\boldsymbol{\mu}$)} \\[5pt]
\Rightarrow \hat{\boldsymbol{\mu}}_{ML} &= \frac{1}{n} \sum_{i=1}^{n} \mathbf{x_i} \hspace{1cm} &\text{($\Sigma$ is PD)}

\end{alignat*}
</div>

###### Solving for the covariance matrix

<blockquote class="math">
<strong>Summary of math used:</strong> use of log rules, the <a class="reference external" href="{{page.url}}#trace_trick">trace trick</a>, more <a class="reference external" href="{{page.url}}#matrix_cook">matrix calculus results</a> and again noting that $\Sigma$ is positive definite (hence invertible) so we can cancel it.
</blockquote>

<div class="math">
\begin{alignat*}{1}

0 &= \nabla_{\Sigma} \sum_{i=1}^{n}-\frac{1}{2} \ln (2 \pi)^{d}|\Sigma|-\frac{1}{2}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)^{T} \Sigma^{-1}\left(\mathbf{x_i}-\boldsymbol{\mu}\right) \hspace{1cm} &\text{(by log rules)} \\[5pt]
&= -\frac{n}{2} \nabla_{\Sigma} \ln |\Sigma|-\frac{1}{2} \nabla_{\Sigma} \operatorname{trace}\left(\Sigma^{-1} \sum_{i=1}^{n}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)\left(\mathbf{x_i}-\boldsymbol{\mu}\right)^{T}\right) \hspace{1cm} &\text{(by trace trick)} \\[5pt]
&= -\frac{n}{2} \Sigma^{-1}+\frac{1}{2} \Sigma^{-2} \sum_{i=1}^{n}\left(\mathbf{x_i}-\boldsymbol{\mu}\right)\left(\mathbf{x_i}-\boldsymbol{\mu}\right)^{T} &\text{(diff. w.r.t $\Sigma$)} \\[5pt]
\Rightarrow \hat{\Sigma}_{ML}&=\frac{1}{n} \sum_{i=1}^{n}\left(\mathbf{x_i}-\hat{\boldsymbol{\mu}}_{ML}\right)\left(\mathbf{x_i}-\hat{\boldsymbol{\mu}}_{ML}\right)^{T} \hspace{1cm} &\text{($\Sigma$ is PD)}

\end{alignat*}
</div>

<a name="mean_var_mle_solution"></a>
##### Mean and variance of MLE solution

The mean calculation follows straight from definitions:

<div class="math">
\begin{alignat*}{1}

\mathbb{E}\left[\mathbf{w}_{ML}\right] &= \mathbb{E}\left[\left(X^{T} X\right)^{-1} X^{T} \mathbf{y}\right]   \hspace{2cm} &\text{(using $\mathbf{w}_{ML} = (X^{T} X)^{-1} X^{T} \mathbf{y}$ )} \\[5pt]
&=\left(X^{T} X\right)^{-1} X^{T} \mathbb{E}[\mathbf{y}] \hspace{2cm} &\text{(only $\mathbf{y}$ is random)} \\[5pt]
&=\left(X^{T} X\right)^{-1} X^{T} X \mathbf{w} &\text{(using $\mathbb{E}[\mathbf{y}] = X \mathbf{w}$)} \\[5pt]
&=\mathbf{w}
\end{alignat*}
</div>

The variance calculation is a little ugly but starts the same way as the mean calculation before using [prob fact 2](#prob_fact2).

<div class="math">
\begin{alignat*}{1}

\operatorname{Var}\left[\mathbf{w}_{ML}\right]
&= \mathbb{E}\left[\mathbf{w}_{ML} \mathbf{w}_{ML}^{T}\right]-\mathbb{E}\left[\mathbf{w}_{ML}\right] \mathbb{E}\left[\mathbf{w}_{ML}\right]^{T}  &\text{(by definition)}  \\[5pt]
&=\mathbb{E}\left[\left(X^{T} X\right)^{-1} X^{T} \mathbf{y} \mathbf{y}^{T} X\left(X^{T} X\right)^{-1}\right]-\mathbf{w} \mathbf{w}^{T}  &\text{($\mathbf{w}_{ML} = (X^{T} X)^{-1} X^{T} \mathbf{y}$)} \\[5pt]
&=\left(X^{T} X\right)^{-1} X^{T} \mathbb{E}\left[\mathbf{y} \mathbf{y}^{T}\right] X\left(X^{T} X\right)^{-1}-\mathbf{w} \mathbf{w}^{T} &\text{(only $\mathbf{y}$ is random)} \\[5pt]
&=\left(X^{T} X\right)^{-1} X^{T}\left(\sigma^{2} I+X \mathbf{w} \mathbf{w}^{T} X^{T}\right) X\left(X^{T} X\right)^{-1}-\mathbf{w} \mathbf{w}^{T} \hspace{0.5cm} &\text{(prob fact 2)} \\[5pt]
&=\left(X^{T} X\right)^{-1} X^{T} \sigma^{2} I X\left(X^{T} X\right)^{-1}+\cdots \\[5pt]
&\left(X^{T} X\right)^{-1} X^{T} X \mathbf{w} \mathbf{w}^{T} X^{T} X\left(X^{T} X\right)^{-1}-\mathbf{w} \mathbf{w}^{T} &\text{(expanding)} \\[5pt]
&=\sigma^{2}\left(X^{T} X\right)^{-1}
\end{alignat*}
</div>


<a name="mean_var_rr_solution"></a>
##### Mean and variance of RR solution

The mean and variance solutions for RR solution follow exactly the same procedure as for the MLE solution and so aren't repeated but left as an exercise.

The solutions are:

$$
\mathbb{E}\left[\mathbf{w}_{RR}\right]=\left(\lambda I+X^{T} X\right)^{-1} X^{T} X \mathbf{w}, \quad \operatorname{Var}\left[\mathbf{w}_{RR}\right]=\sigma^{2} Z\left(X^{T} X\right)^{-1} Z^{T}
$$

where $ Z=\left(I+\lambda\left(X^{T} X\right)^{-1}\right)^{-1}.$

<a name="svd_ols_rr"></a>
##### Analysis of OLS and RR using SVD

The solutions to least squares and ridge regression are very similar, and it is possible to use the [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to analyse the difference between $\mathbf{w}\_{LS}$ and $\mathbf{w}\_{RR}$.

In particular we write the data matrix $X$ as $X = USV^T$ with:
<div class="bullet"> 
<li> $U$: $n \times d$ with orthonormal columns </li>
<li> $S$: $d \times d$ with non-negative diagonal entries </li>
<li> $V$: $d \times d$ with orthonormal columns</li>
</div>
<br>
We give just a sketch summary of the result here and the reader is referred to [ESL](#esl) [3.4.1] for more details.

The upshot is that the regularization term $\lambda$ in the ridge regression solution acts as a sort of protection to stop us dividing by really small values when calculating $\mathbf{w}\_{RR}$. This is loosely analogous to what we sometimes do to avoid division by 0 errors or blow-ups by adding a term, $\epsilon$, to the denominator $\frac{1}{x + \epsilon}$.

In particular it is shown that:

$$\mathbf{w}_{RR} = VS_{\lambda}^{-1}U^T \mathbf{y}$$

where:
<div class="bullet"> 
<li> $S$ is a matrix holding the singular values (i.e. the square roots of the eigenvalues) of $X$ </li>
<li> $S_{\lambda}^{-1}$ refers to a diagonal matrix with each term of the form:

$$\dfrac{S_{ii}}{\lambda + S^2_{ii}}$$
</li>
<li> $S$ is a $d$ by $d$ matrix </li>
</div>
<br>
So we can see that $\lambda$ in the SVD view of $\mathbf{w}\_{RR}$ above stops us getting weights that are huge when we have small singular values in $X$ - no such protection exists for the OLS solution ($\lambda = 0$).

<hr class="with-margin">
