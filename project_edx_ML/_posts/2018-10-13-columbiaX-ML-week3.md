---
layout: post
title: ColumbiaX - ML - week 3
date: 2018-10-13
use_math: true
image: "bayes_lr.png"
comments: true
tags: [bayesian linear regression, active learning, sparse regression]
---
We move linear regression into a fully Bayesian setting as well as introducing predictive distributions and active learning. We then finish the topic of regression by looking at how to achieve sparse solutions through lasso regression.
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

[Bayesian linear regression](#blr) naturally extends the MAP solution from ridge regression, which is a point esimate, to obtain a full probability distribution of the model parameters.

This leads onto a discussion about [posterior predictive distributions](#pred_dist) which allow us to get uncertainty estimates for a prediction. There are many ways to use uncertainty estimates from a Bayesian analysis and [active learning](#active) is discussed as one such application.

We then finish by discussing [sparse regression](#sparse) to address cases where the dimensionality of the problem is much larger than the number of samples.

<a name="blr"></a>
<hr class="with-margin">
<h4 class="header" id="blr">Bayesian linear regression</h4>

<blockquote class="tip">
<strong>Summary:</strong> Bayesian linear regression allows us to obtain a full probability distribution for the model parameters.
</blockquote>

##### Motivation

Bayesian inference is cool as it allows us to characterize uncertainty about the parameters $\mathbf{w}$ using Bayes' rule as well as providing a principled framework for explicitly incorporating knowledge via the prior.

This opens up a range of possibilities, for instance, we can now ask questions such as how different is a given parameter $w_i$ from 0 (i.e. is it significant or not?). This is a departure from the non-Bayesian setting where we had limited ways to answer such questions. This is a very practical and useful application of Bayesian regression and occurs commonly in many fields. For example, if our variables are factors in a medical trial and we wish to determine whether the factor under scrutiny is actually having an impact (before we spend money on it) it helps to know if the distribution of this parameter has a big variance vs. say, being tightly distributed and centred away from 0.

##### Calculating the posterior

<blockquote class="tip">
<strong>Background</strong>
<br>
Before we start it helps to recall Bayes' rule:

$$
p(\mathbf{w} | \mathbf{y}, X) = \dfrac{p(\mathbf{y} | \mathbf{w}, X) p(\mathbf{w})}{p(\mathbf{y} | X)}
$$

where for Bayesian linear regression we assume:
<br>
<br>
Likelihood: $\mathbf{y} \sim \mathcal{N}\left(X \mathbf{w}, \sigma^{2} I\right)$
<br>
Prior: $\mathbf{w} \sim \mathcal{N}\left(0, \lambda^{-1} I\right).$
<br>
<br>
In
<a class="reference external" href="/../project_edx_ml/2018/10/10/columbiaX-ML-week1and2">weeks 1 and 2</a>  we considered linear regression from both a maximum likelihood (ML) and MAP perspective.
<br>
<br>
We can now summarise the 3 approaches:
<br>
<br>
<strong>ML</strong>: point estimate using likelihood only, $p(\mathbf{y} | \mathbf{w}, X)$
<br>
<strong>MAP</strong>: point estimate using likelihood and prior, $p(\mathbf{y} | \mathbf{w}, X)\, p(\mathbf{w})$
<br>
<strong>Bayesian linear regression</strong>: full posterior distribution, $p(\mathbf{w} | \mathbf{y}, X)$
</blockquote>

In order to calculate the posterior we need to calculate $p(\mathbf{y} \| X)$, but what exactly is it?

Recall we have assumed a probabilistic model for the data which is the numerator in Bayes' rule as the likelihood multiplied by the prior belief of the parameters. In the numerator we evaluate this likelihood for a given set of parameters e.g. calculate $p(\mathbf{y} \| \mathbf{w}, X)$.

For the denominator we need to calculate the probability of the data independent of any parameters - this ensures the shape of the posterior is solely due to the numerator and not the denominator, which is really just a normalising constant.

We don't actually know what the probability of the data independent of any parameters is and so the tactic used to calculate $p(\mathbf{y} \| X)$ is to take the numerator (which contains the model assumptions) and integrate out any parameters. We do this by noting that the denominator can be expressed as:

$$
p(\mathbf{y} | X) = \int_{\mathbb{R}^{d}} p(\mathbf{y} | \mathbf{w}, X) \, p(\mathbf{w}) \, d\mathbf{w}
$$

Unfortunately this term $p(\mathbf{y} \| X)$ is usually not calculable in Bayesian analysis due to [the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). However, for Bayesian linear regression under the assumptions we have made for the likelihood and prior we are able to calculate an analytic solution.

It is [shown](#post_blr) in the appendix that the posterior is:

$$
p(\mathbf{w} | \mathbf{y}, X) = \mathcal{N}(\mathbf{w} | \boldsymbol{\mu}, \Sigma)
$$

with

<a name="mu_sig_post"></a>

$$ \boldsymbol{\mu} = (\lambda \sigma^{2} I + X^T X)^{-1} X^T \mathbf{y} \,\, , \,\, \Sigma = (\lambda I + \sigma^{-2} X^T X)^{-1}.$$

And so the posterior distribution is also a Gaussian distribution with dimensionality equal to the number of parameters in $\mathbf{w}$.

<blockquote class="tip">
<strong>Note:</strong> we notice that $\boldsymbol{\mu} = \mathbf{w}_{MAP}.$
<br>
<br>
Thus Bayesian linear regression centres the posterior around the MAP solution from ridge regression except now we have a full probability distribution for the parameters, $p(\mathbf{w} | \mathbf{y}, X)$.
</blockquote>

##### Plotting Bayesian linear regression

In Bayesian linear regression we return a full distribution for each parameter which is a Gaussian distribution. As such we can sample from this distribution and visualise the distribution of competing hypotheses for the best fit line. Each sample from the posterior is a slope and intercept term and we plot its line in light blue - we draw 100 samples. The OLS solution is shown in red (note this isn't the ridge regression solution but in this example will be very similar).

Note, as the below fit was done with [scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_bayesian_ridge.html) there's a little more going on under the hood in terms of hyperparameter estimation, but this doesn't change the story - we still get 'many' solutions from a Bayesian linear regression model, not just a single explanation.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/blr_ridge_ols.png" alt="Image" width="600" height="450" />
</p>
<em class="figure">Samples from the Bayesian posterior vs. OLS (red) solution</em>
<hr class="with-margin">

##### Python code

Example python code for the above plot.

<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression

# 1. Generating simulated data
np.random.seed(0)
n_samples, n_feats = 100, 1
X = np.concatenate((np.ones((n_samples, 1)), np.random.randn(n_samples, n_feats)), 1)
w = np.array([3, 0.7])  # random weight
alpha_ = 1  # create noise with a precision of alpha
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
y = np.dot(X, w) + noise  # create target

# 2. Fit the Bayesian ridge regression and an OLS for comparison
blr = BayesianRidge(compute_score=True, fit_intercept=False)
blr.fit(X, y)
ols = LinearRegression(fit_intercept=False)
ols.fit(X, y)

# 3. Sampling
x_min, x_max = np.min(X[:, 1]) - 0.25, np.max(X[:, 1]) + 0.25
x_grid = np.expand_dims(np.linspace(x_min, x_max, 100), 1)
x_test = np.concatenate((np.ones((n_samples, 1)), x_grid), 1)
samples = np.random.multivariate_normal(blr.coef_, blr.sigma_, size=100)
y_pred = ols.predict(x_test)

# 4. Plotting
plt.scatter(X[:, 1], y, color = 'green')
for i in range(samples.shape[0]):
	y_tmp_pred = np.dot(x_test, samples[i, :])
	plt.plot([x_test[0, 1], x_test[-1, 1]], [y_tmp_pred[0], y_tmp_pred[-1]],
            alpha=0.25, color='skyblue')
plt.plot([x_test[0, 1], x_test[-1, 1]], [y_pred[0], y_pred[-1]], 'k-', color='red')
plt.title('Bayesian linear regression vs. OLS')
plt.show()
</code></pre>

<a name="pred_dist"></a>
<hr class="with-margin">
<h4 class="header" id="pred_dist">Predicting new data</h4>

<blockquote class="tip">
<strong>Summary:</strong> the posterior predictive distribution for an unseen data-point allows us to get an estimate of the uncertainty associated with the prediction in the form of a full probability distribution.
<br>
<br>
This distribution is sometimes called the posterior predictive distribution (or just predictive distribution) and is a powerful feature of Bayesian analysis as we will see later for
<a class="reference external" href="{{page.url}}#active">active learning</a>.
</blockquote>

##### Introduction

Having learned a posterior distribution for the model's weights Bayesian linear regression allows us to also make probabilistic statements about new predictions. This means that for any new data point $(\mathbf{x}_0, y_0)$ we are able to calculate a full probability distribution for the predicted value of $\mathbf{x}_0$. This is done without reference to the unknown test value $y_0$ and we denote this distribution as $p(y_0 \| \mathbf{x}_0, \mathbf{w})$.

This deviates from what is done in, say, linear or ridge regression where we simply return a point prediction equal to $\mathbf{x}_0^T \mathbf{w}\_{LS}$ or $\mathbf{x}_0^T \mathbf{w}\_{RR}$.

It turns out that in the context of Bayesian linear regression we are able to calculate the posterior predictive distribution exactly.

##### Posterior predictive distribution

The starting point for understanding the posterior predictive distribution is to recall that we will have already performed some model fitting from data and already have the posterior distribution of the model's weights. That is, we don't need the original data set $(X, \mathbf{y})$ which the original model was fit on, just the posterior distribution $p(\mathbf{w} \| \mathbf{y}, X)$.

The posterior predictive distribution can be written as:

<div class="math">
\begin{alignat*}{1}

p(y_{0} | \mathbf{x}_0, \mathbf{y}, X) &= \int_{\mathbb{R}^{d}} p(y_{0}, \mathbf{w} | \mathbf{x}_0, \mathbf{y}, X) \, d\mathbf{w}
   \\[5pt]
&= \int_{\mathbb{R}^{d}} p(y_{0} | \mathbf{w}, \mathbf{x}_0, \mathbf{y}, X) \, p(\mathbf{w} | \mathbf{x}_0, \mathbf{y}, X) \, d\mathbf{w} \\[5pt]
&= \int_{\mathbb{R}^{d}} \underbrace{p(y_{0} | \mathbf{w}, \mathbf{x}_0)}_\text{likelihood} \,  \underbrace{p(\mathbf{w} | \mathbf{y}, X)}_\text{posterior} \, d\mathbf{w}. \hspace{1cm} &\text{(by cond. indep.)} 
\end{alignat*}
</div>

Intuitively this is saying that we evaluate the likelihood for a new prediction $y_0$ given an observed $\mathbf{x}_0$ and some values for $\mathbf{w}$ and then weight this by the current belief we have for $\mathbf{w}$ given the original data we trained the model on, $(X, \mathbf{y})$. The best current belief we have about the model's parameters is precisely the posterior distribution $p(\mathbf{w} \| \mathbf{y}, X)$.

We then integrate over all possible values of $\mathbf{w}$. This integrating over all values of $\mathbf{w}$ is what allows us to obtain a probability distribution for the prediction - we are simultaneously considering all values that $\mathbf{w}$ could take from the posterior distribution and for each possible set of $\mathbf{w}$ values, we calculate the likelihood using them and then weight the likelihood by the probability of those values occurring.

##### Predictive equations

In order to obtain the predictive equations for the posterior predictive distribution, $p(y_0 \| \mathbf{x}_0, \mathbf{w})$, we first recall the form of the likelihood and posterior from above$:


$$
\underbrace{p\left(y_{0} | \mathbf{x}_0, \mathbf{w}\right) = \mathcal{N}\left(y_{0} | \mathbf{x}_0^{T} \mathbf{w}, \sigma^{2}\right)}_\text{likelihood}, \, \,
\underbrace{p(\mathbf{w} | \mathbf{y}, X) = \mathcal{N}(\mathbf{w} | \boldsymbol{\mu}, \Sigma)}_\text{posterior}
$$

where we know the values of $\boldsymbol{\mu}$ and $\Sigma)$ from [above](#mu_sig_post).

Given the above assumptions the posterior predictive distribution for a new test point $\mathbf{x}_0$ is also a Gaussian:

<a name="pred_eqtns"></a>
<div class="math">
\begin{alignat*}{1}

p(y_{0} | \mathbf{x}_0, \mathbf{y}, X) &=  \mathcal{N}(y_0 | \mu_0, \sigma_0^2)  \\[5pt]
\mu_0 &= \mathbf{x}_0^{T} \boldsymbol{\mu} \\[5pt]
\sigma_0^2 &= \sigma^2 + \mathbf{x}_0^T \Sigma \mathbf{x}_0. 
\end{alignat*}
</div>

We can see that the expected value of the prediction we make is the same as for the MAP solution (recall $\boldsymbol{\mu} = (\lambda \sigma^{2} I + X^T X)^{-1} X^T \mathbf{y}$) but now we are able to quantify the uncertainty in the prediction with $\sigma_0^2$.

<blockquote class="tip">
<strong>Sidebar on $\sigma^2$</strong>
<br>
In the last two posts we have derived results based upon the assumption we made about the likelihood:

$$\mathbf{y} \sim \mathcal{N}\left(X \mathbf{w}, \sigma^{2} I\right)$$

without talking about how to estimate $\sigma^2$. It's actually simple to do this as the solution to $\mathbf{w}_{ML}$ doesn't depend on $\sigma^{2}$ and recalling that the prediction we make from maximum likelihood is $X\mathbf{w}_{ML}$.
<br>
<br>
We can thus calculate $\sigma^{2}$ as:

$$
\sigma^{2} = \frac{1}{n} \sum_{i=1}^{n}(y_i - \mathbf{x}_i \mathbf{w}_{ML})^2
$$

which is essentially the sample variance calculated for the residuals.
</blockquote>

<a name="active"></a>
<hr class="with-margin">
<h4 class="header" id="active">Active learning</h4>

##### Background

We switch now to think again about the Bayesian posterior for the model parameters, $p(\mathbf{w} \| \mathbf{y}, X)$. A situation that often arises in practical applications is that a model is fitted to some initial existing data and then we have some freedom to choose which data-point, $\mathbf{x}_0$, to query next in order to obtain a label $y_0$. For example, it may be the case that we can choose which experiment to conduct next, where to drill for oil, which advert to run next etc...

Given we have this choice, and that procuring a labelled data-point may be a costly procedure it would be nice if we had some rigorous way to decide which $\mathbf{x}_0$ to choose next and then use it to update the current posterior we have.

Helpfully, Bayesian learning can be thought of as a sequential process whereby the posterior after seeing some data becomes the prior for the next set of data we receive. This continuous ability to update the beliefs we have in light of new data in a principled manner is a key feature of Bayesian analysis.

Active learning takes advantage of this feature of Bayesian analysis and for the case of linear regression all solutions can be obtained in closed form.

##### Active learning

<blockquote class="tip">
<strong>Summary:</strong> active learning is where we have a choice as to which data-point to observe the output for next and we then use this output to sequentially update the model.
<br>
<br>
<strong>Question:</strong> can we choose which data-point to measure next in order to update the model posterior intelligently?
</blockquote>

There are many algorithms which offer a solution to the question of which data-point to measure next. We will discuss one such algorithm which will also answer the question of what we mean by intelligently in the above box.

We need to do two things:

<div class="bullet"> 
<ol> 1. Find a way to sequentially update the parameters of the posterior</ol>
<ol> 2. Determine a strategy for choosing which data-point to measure next</ol>
</div>

We will do the above in the context of Bayesian linear regression.

##### Sequentially updating the posterior
In order to sequentially learn the posterior distribution we need a way to sequentially learn the parameter estimates $\boldsymbol{\mu}$ and $\Sigma$ for the Bayesian linear regression posterior. Recall the posterior is given by:

$$
p(\mathbf{w} | \mathbf{y}, X) = \mathcal{N}(\mathbf{w} | \boldsymbol{\mu}, \Sigma)
$$

with

$$ \boldsymbol{\mu} = (\lambda \sigma^{2} I + \underbrace{X^T X}_\text{$(d \times n) \times (n \times d)$})^{-1} \underbrace{X^T \mathbf{y}}_\text{$(d \times n) \times (n \times 1)$} \,\, , \,\, \Sigma = (\lambda I + \sigma^{-2} \underbrace{X^T X}_\text{$(d \times n) \times (n \times d)$})^{-1}$$

where in the above we draw attention to the dimensions of the parameters $\boldsymbol{\mu}$ and $\Sigma$. This is to highlight that their final dimensionality does not involve $n$, the number of observations. This, and the fact that matrix multiplication can be written in terms of [outer products](#mmult_add) in an additive manner means we are able to decompose the parameter update into 2 steps:

<div class="bullet"> 
<ol> 1. Calculate the parameter estimates for $\boldsymbol{\mu}$ and $\Sigma$ with the original data</ol>
<ol> 2. Update $\boldsymbol{\mu}$ and $\Sigma$ sequentially in an additive manner using the new data</ol>
</div>

We are thus able to re-express the above definitions of $\boldsymbol{\mu}$ and $\Sigma$ to split them into terms based on original data and terms based on new data. The parameter update equations are:

<div class="math">
\begin{alignat*}{1}

p(\mathbf{w} | y_0, \mathbf{x}_0, \mathbf{y}, X) &= \mathcal{N}(\mathbf{w} | \boldsymbol{\mu}, \Sigma) \\[5pt]

\boldsymbol{\mu} &= (\lambda \sigma^{2} I+(\overbrace{\mathbf{x}_0 \mathbf{x}_0^{T}}^\text{new data}+ \overbrace{\underbrace{\sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^{T}}_\text{$= \, X^T X$}}^\text{original data})^{-1}(\overbrace{\mathbf{x}_0 y_{0}}^\text{new data}+ \overbrace{\underbrace{\sum_{i=1}^{n} \mathbf{x}_i y_{i}}_\text{$= \, X^T \mathbf{y}$}}^\text{original data})) \\[5pt]

\Sigma &= (\lambda I+\sigma^{-2}(\overbrace{\mathbf{x}_0 \mathbf{x}_0^{T}}^\text{new data}+\overbrace{\underbrace{\sum_{i=1}^{n} \mathbf{x}_i \mathbf{x}_i^{T}}_\text{$= \, X^T X$}}^\text{original data}))^{-1}
\end{alignat*}
</div>

The above works because matrix multiplication can be written in terms of [outer products](#mmult_add) and can be computed additively. The reader is encouraged to convince themselves of the above before proceeding.

Now we have a way to update the posterior parameters in light of new data we turn to determining how choose which data-point to measure next.

##### An active learning strategy

<blockquote class="algo">
<hr class="small-margin">
<strong>Algorithm: an active learning strategy (posterior entropy minimization)</strong>
<hr class="small-margin">
We fit a model to the original data $(\mathbf{y}, X)$ to get the posterior, $p(\mathbf{w} | \mathbf{y}, X)$. We then proceed as follows:
<br>
<br>
1. Calculate the posterior predictive distribution, $p(y_0 | \mathbf{x}_0, \mathbf{w})$, for every $\mathbf{x}_0$ we are considering measuring
<br>
2. Choose the $\mathbf{x}_0$ for which $\sigma_0^2$ is largest and measure $y_0$
<br>
3. Update the posterior with the new data-point using the sequential posterior parameter update equations
<br>
4. Return to step 1 using the updated posterior
<br>
<br>
Recall the posterior predictive equations from <a class="reference external" href="{{page.url}}#pred_eqtns">above</a>.
</blockquote>

Intuitively the above algorithm is choosing the point, $\mathbf{x}_0$, where we are most uncertain about the predicted value. This is a natural thing to want to do, if we already have data corresponding to a certain region of the input space we are interested in it perhaps doesn't make sense to keep querying the same region.

As an interesting note, it can be shown that the above strategy is minimizing the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the posterior distribution in a greedy fashion.

##### What active learning is not

Active learning is not [online learning](https://en.wikipedia.org/wiki/Online_machine_learning) which is similar to active learning except we don't have a choice about which new data we receive. In online learning new data simply becomes available in a sequential order and we must use it to update the model at each step.

<a name="sparse"></a>
<hr class="with-margin">
<h4 class="header" id="sparse">Sparse regression</h4>

##### Motivation

Many modern data-sets are high-dimensional ($d \gg n$) and often a lot of these features are irrelevant for predicting the target, $\mathbf{y}$. For example, in genomic data we can have 1000's of genes but far fewer subjects.

Ideally we would like some way to perform feature selection in a way that is learned from the data. In this section we will introduce different penalty terms that can be added to a loss function as well as analysing the type of solutions they encourage.

<blockquote class="tip">
<strong>Goal:</strong> find a penalty that encourages sparse solutions, that is, sets most entries of the learned weight vector $\mathbf{w}$ to 0. We would like to keep only a few (potentially large) entries of $\mathbf{w}$ which are of most importance in predicting $\mathbf{y}$.
</blockquote>

##### General optimization set-up

Quite often in machine learning we can think of an optimization function loss as composing of two parts, a term which measures how well we are fitting the data and a penalty term. This can be written as:

$$
\mathcal{L}=\underbrace{\sum_{i=1}^{n}\left(y_{i}-f\left(\mathbf{x}_{i}; \mathbf{w}\right)\right)^{2}}_\text{goodness of fit term} + \underbrace{\lambda\| \mathbf{w}\|^{2}}_\text{penalty term}
$$

where the goal is to minimize $\mathcal{L}$. $f$ is a function that is predicting on the data with parameters $\mathbf{w}$ and we are using a sum of squares goodness of fit term. For ridge regression we made the prediction as $f(\mathbf{x}_{i} ; \mathbf{w}) = \mathbf{x}\_{i}^T \mathbf{w}$ and $\lambda \|\| \mathbf{w} \|\|^{2}$ is called a quadratic penalty term.

##### Quadratic penalties

We can compare a quadratic penalty term as for ridge regression to a linear one based on the absolute values of the weights as for lasso regression, which uses $\lambda \|\| \mathbf{w} \|\|$ as a penalty term.

The below chart shows that for a given change in a weight entry $w_j$ the increase in penalty is much larger for the quadratic than linear penalty. This means that reducing a large entry $w_j$ in $\mathbf{w}$ will achieve a reduction in $\mathcal{L}$ that changes depending on the magnitude of $w_j$. This is not the case for the linear penalty whereby the reduction in $\mathcal{L}$ does not depend on the magnitude of $w_j$.

Following the above we reason that quadratic penalties end up favouring solutions for which each entry of $\mathbf{w}$, $w_j$, is of a similar size. Any $w_j$ that gets too large is penalized heavily, even if it is of high predictive importance, and so a quadratic penalty encourages many small but non-zero entries in $\mathbf{w}$.

To achieve sparsity we turn instead to linear penalties, which, faced with a few large entries in $\mathbf{w}$ can set the other entries of $\mathbf{w}$ to 0 to achieve the same reduction in $\mathcal{L}$. This is because a linear penalty doesn't prefer to reduce large $w_j$ over smaller $w_j$ entries.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/penalties_ridge_lasso.png" alt="Image" width="600" height="300" />
</p>
<em class="figure">Quadratic (left) vs. linear (right) penalties <br> Image credit: [edX ColumbiaX ML course](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4)</em>
<hr class="with-margin">

##### Linear penalties

To get an intuition about how linear penalties encourage sparsity a common chart shown is the one below. In order to understand this it is helpful to [write](#ridge_obj) the objective term for ridge regression (we can rewrite the lasso objective term similarly) as:

$$
\|\mathbf{y} - X \mathbf{w} \|^{2}+\lambda\|\mathbf{w}\|^{2} = \underbrace{\left(\mathbf{w}-\mathbf{w}_{LS}\right)^{T}\left(X^{T} X\right)\left(\mathbf{w}-\mathbf{w}_{LS}\right)}_\text{level sets in red} + \underbrace{\lambda \mathbf{w}^{T} \mathbf{w}}_\text{level sets in blue} + \mathrm{const.} \text { w.r.t. } \mathbf{w}
$$

where we are using the idea of level sets - set of points in the domain of a function where the function is constant. Points where both the red and blue curves intersect are viable solutions to the optimization problem - it is worth spending time to understand these charts are they give strong intuition into the objective function.

The geometric reasoning behind why linear penalties encourage sparsity is that due to the shape of the level sets for the linear penalty term, the points of intersection with the red level set terms are more likely to lie along an axis where any given $w_j$ may equal 0. This is due to the 'pointy' nature of the linear penalty level sets. This concept, applied to higher-dimensional $\mathbf{w}$, is an explanation as to how linear penalties encourage sparse solutions.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/ridge_lasso.png" alt="Image" width="600" height="300" />
</p>
<em class="figure">Level sets for lasso (left) and ridge (right) objective terms
<br> Image credit: [edX ColumbiaX ML course](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4)</em>
<hr class="with-margin">

<blockquote class="tip">
<strong>Understanding the role of $\lambda$:</strong> for further understanding of the role $\lambda$ plays, with reference to the above chart, we can imagine that large $\lambda$ pulls the possible solutions (the red dots) towards the point $(w_1, w_2) = (0, 0)$ and far away from the $\mathbf{w}_{LS}$ solution. Equally, small $\lambda$ will permit solutions close to the point $\mathbf{w}_{LS}$. In this sense we can imagine that varying $\lambda$ moves the possible solutions between the point at $(w_1, w_2) = (0, 0)$ and $\mathbf{w}_{LS}$.
</blockquote>

<a name="lp_norm"></a>
##### $l_p$ penalty terms

The ideas of quadratic and linear penalty terms can be generalized to any power $p$ of penalty, called the [$l_p$ norm](https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm) of a vector. That is, we can consider penalty terms of the form $\lambda \|\| \mathbf{w} \|\|^p$ where:

$$
|| \mathbf{w} ||^p = \left(\sum_{j=1}^{d}\left|w_{j}\right|^{p}\right)^{\frac{1}{p}} \hspace{2cm} \text{for $0<p \leq \infty$}
$$

The norm of a vector in a loose sense can be thought of as some measure of its length and hence cannot be negative.

Depending on the value of $p$ we obtain different shapes for the level sets of the penalty term. The chart below shows this for some choices of $p$.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/lp_norms.png" alt="Image" width="600" height="150" />
</p>
<em class="figure">Shape of penalty term level sets for different values of $p$ <br> Image credit: [edX ColumbiaX ML course](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4)</em>
<hr class="with-margin">

##### Impact of $p$ on computational tractability

We note the following facts (using a sum of squares loss) as we vary $p$:
<div class="bullet"> 
<li> $p < 1$: we can only find approximate solutions to the optimization problem using iterative algorithms </li>
<li> $p \geq 1$, $p \neq 2$: convex optimization problem able to be solved exactly (includes lasso regression) </li>
<li> $p=2$: closed form solution (ridge regression) </li>
</div>
<br>
The loss of convexity in the optimization problem for $p < 1$ can be seen be noting that in the above chart we lose 'line of sight' between all points in the level sets of the penalty term.

<a name="math_details_sec"></a>
<hr class="with-margin">
<h4 class="header" id="math_results">Some mathematical results</h4>

<a name="mmult_add"></a>
##### Matrix multiplication as outer products

To see that we can compute matrix multiplication sequentially consider a matrix $A$ with dimensions $3 \times 2$ for which we wish to compute $A^T A$. We call a row of $A$, $\mathbf{a}_i$.

Let's say $A$ is given by:

$$
A =
\left[
\begin{array}{cc}
{1} & {3}\\
{2} & {1}\\
{3} & {4}
\end{array}
\right]
$$

and so

$$
A^T A =
\left[
\begin{array}{cc}
{14} & {17} \\
{17} & {26} \\
\end{array}
\right]
$$

This is exactly the same as performing:

<div class="math">
\begin{alignat*}{1}
A^T A &=  \mathbf{a}_1^T \mathbf{a}_1+\mathbf{a}_2^T \mathbf{a}_2+\mathbf{a}_3^T \mathbf{a}_3 \\[5pt]
&= \sum_{i=1}^{n=3} \mathbf{a}_i\mathbf{a}_i^T
\end{alignat*}
</div>

where we have rewritten matrix muliplication in terms of outer products. The above results holds in [general](https://en.wikipedia.org/wiki/Outer_product#Definition_(matrix_multiplication)) and for Bayesian linear regression means we are able to update the posterior parameter equations as new data comes in by simply computing the outer product of the new data, $\mathbf{x}_0\mathbf{x}_0^T$, and add it to the existing estimate we have for $X^T X$.

Similar reasoning allows $X^T \mathbf{y}$ to be updated in the same way.

<hr class="with-margin">
<h4 class="header" id="references">References</h4>

<div class="bullet"> 
<li>
<a name="prml"></a>
Bishop, C. (2006). Chapters: 3.3 - 3.5; <a class="reference external" href="https://www.springer.com/gb/book/9780387310732">Pattern Recognition and Machine Learning</a>.</li>
<li>
<a name="esl"></a>
Hastie, T., R. Tibshirani, and J. Friedman (2001). Chapters: 3.3 - 3.8;  <a class="reference external" href="http://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a>.</li>
<li>
<a name="edx_ml"></a>
edX, ColumbiaX, <a class="reference external" href="https://www.edx.org/course/machine-learning-1">Machine Learning</a>.</li>
</div>

<hr class="with-margin">
<h4 class="header" id="appendix">Appendix</h4>

<a name="post_blr"></a>
##### Posterior for Bayesian linear regression

Here we derive the posterior for Bayesian linear regression and show it is a Gaussian distribution.

<blockquote class="tip">
<strong>Target form for posterior</strong>
<br>
The posterior we are aiming for will have the form:

<div class="math">
\begin{alignat*}{1}
p(\mathbf{w} | \boldsymbol{\mu}, \Sigma) &= \frac{1}{(2 \pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left( -\frac{1}{2}(\mathbf{w}-\boldsymbol{\mu})^{T} \Sigma^{-1}(\mathbf{w}-\boldsymbol{\mu}) \right) \\[5pt]
&= \frac{1}{(2 \pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}}} \exp \left( -\frac{1}{2} \left( \underbrace{\mathbf{w}^{T} \Sigma^{-1} \mathbf{w}}_\text{quadratic in $\mathbf{w}$} - \underbrace{2 \mathbf{w}^{T} \Sigma^{-1} \boldsymbol{\mu}}_\text{linear in $\mathbf{w}$} + \underbrace{\boldsymbol{\mu}^{T} \Sigma^{-1} \boldsymbol{\mu}}_\text{no $\mathbf{w}$ dep.} \right) \right) \hspace{1cm} &\text{(A1)}
\end{alignat*}
<br>
for some $\boldsymbol{\mu}$ and $\Sigma$.
</div>
</blockquote>

We start by working with proportionality of the likelihood and prior and then will deduce the normalising constant by [completing the square](https://en.wikipedia.org/wiki/Completing_the_square) which is a common technique used in Bayesian analysis when trying to obtain a posterior form that is a known distribution.

<div class="math">
\begin{alignat*}{1}
p(\mathbf{w} | \mathbf{y}, X) &\propto p(\mathbf{y} | \mathbf{w}, X) \, p(\mathbf{w}) \\[5pt]

&\propto \exp \left({-\frac{1}{2 \sigma^{2}}(\mathbf{y}-X \mathbf{w})^{T}(\mathbf{y}-X \mathbf{w})}\right) \exp \left({-\frac{\lambda}{2} \mathbf{w}^{T} \mathbf{w}}\right) \hspace{0.25cm} &\text{by defn.} \\[5pt]

&\propto \exp \left( {-\frac{1}{2 \sigma^{2}} \left[
\underbrace{\mathbf{y}^T \mathbf{y}}_\text{no $\mathbf{w}$ dep.} \underbrace{ - \, \mathbf{y}^T X \mathbf{w} - \mathbf{w}^T X^T \mathbf{y}}_\text{$1 \times 1 \, = \, 2 \, \mathbf{w}^T X^T \mathbf{y}$} + \mathbf{w}^T X^T X \mathbf{w} \right] -\frac{\lambda}{2} \mathbf{w}^T \mathbf{w}} \right)  \hspace{0.25cm} &\text{expand} \\[5pt]

&\propto \exp \left( -\frac{1}{2} \left( \underbrace{\mathbf{w}^{T}\left(\lambda I+\sigma^{-2} X^{T} X\right) \mathbf{w}}_\text{quadratic in $\mathbf{w}$} - \underbrace{2 \sigma^{-2} \mathbf{w}^{T} X^{T} \mathbf{y}}_\text{linear in $\mathbf{w}$} \right) \right) &\text{(A2)}
\end{alignat*}
</div>

The form of the above allows us to conclude the posterior will have a Gaussian distribution as we have matched the form that $\mathbf{w}$ appears in (A1).

Due to proportionality we are able to multiply and divide by any terms that don't depend on $\mathbf{w}$, which we did above. In order to make the above exact we need to find the normalising constant, $Z$, such that we end up with a form as in (A1).

Comparing the terms from (A1) and (A2) that are quadratic and linear in $\mathbf{w}$ we might be tempted to set:

$$
\Sigma^{-1}=\left(\lambda I+\sigma^{-2} X^{T} X \right), \, \, \,
\Sigma^{-1} \boldsymbol{\mu} = X^{T} \mathbf{y} \sigma^{-2} \tag{A3}
$$

in (A2). If we did this we only then need to decide what the normalising constant, $Z$, should be in order to match (A1) - and we can set it to anything we like as long as it doesn't involve $\mathbf{w}$.

Looking at (A1) to see what we are missing in (A2) implies we set:

$$
Z=(2 \pi)^{\frac{d}{2}}|\Sigma|^{\frac{1}{2}} \underbrace{\exp ({\frac{1}{2} \boldsymbol{\mu}^{T} \Sigma^{-1}  \boldsymbol{\mu}}}_\text{missing in (A2)}) \tag{A4}
$$

The above leads directly to the Bayesian posterior as

$$
p(\mathbf{w} | \mathbf{y}, X) = \mathcal{N}(\mathbf{w} | \boldsymbol{\mu}, \Sigma)
$$

with

$$ \boldsymbol{\mu} = (\lambda \sigma^{2} I + X^T X)^{-1} X^T \mathbf{y}, \,\,\,\, \Sigma = (\lambda I + \sigma^{-2} X^T X)^{-1} \tag{A5}

$$

where we manipulate $\boldsymbol{\mu}$ from (A3) slightly to get the above form.

<a name="ridge_obj"></a>
##### Different form for ridge regression objective term

Simply expand:

$$
\left(\mathbf{w}-\mathbf{w}_{LS}\right)^{T}\left(X^{T} X\right)\left(\mathbf{w}-\mathbf{w}_{LS}\right)
$$

using:

$$
\mathbf{w}_{LS} = (X^T X)X^T \mathbf{y}
$$

and fact that $X^T X$ is symmetric.

<hr class="with-margin">
