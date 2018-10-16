---
layout: post
title: ColumbiaX - Machine Learning - week 3
date: 2018-10-13
use_math: true
---

This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.

## Week 3 (lectures 5 and 6): overview

Week 3 brought the topic of linear regression to an end and whilst there were some interesting ideas in the lectures there was also a few trickier aspects which I hope to be able to largely clarify below (for my own understanding as much as anything). First, the main ideas:

* Bayesian Linear Regression
  * Here we really just extend MAP (a point estimate) to obtain a full probability distribution of our parameters, $w$.
* Predicting new data with Bayes
  * We are now able to calculate the probability of a new target observation, $y_0$ given our original training set $(y, X)$ and a new test example $x_0$.
* Active learning
  * Updating our model's parameters sequentially as new data comes in. The key point is that our current posterior (of the model's parameters) before the new data becomes our prior in the presence of new data.
* Entropy minimization
  * Given freedom over which new test target $y_i$ to measure we devise a strategy which is shown to be minimizing the uncertainty in our posterior.
* Lasso Linear Regression
  * Often when $d \gg n$ we wish to perform feature selection with our data. The Lasso is analogous to Ridge Regression except we change the penalty on our weights to be the sum of the absolute values of the individual weights.



## Week 3 (lectures 5 and 6): the big picture

Week 3 brings the way we look at linear regression into a fully Bayesian framework before moving on to talk about the case where we have more dimensions to our data than observations. Bayesian Linear Regression allows us to ask (and answer!) questions with our data we are unable to do in a non-Bayesian setting.

#### Bayesian Linear Regression allows us to obtain a full probability distribution for our model parameters

We can now ask questions such as whether a given parameter $w_i$ is different from 0 (i.e. whether it is significant or not). This is a departure from the non-Bayesian setting where we had no way to answer such questions. This is a very practical and useful application of Bayesian regression and occurs commonly in many fields. For example, if our variables are factors in a medical trial and we wish to determine whether the factor under scrutiny is actually having an impact (before we spend anymore money on it) it helps to know if the distribution of this parameter has a big variance vs. say, being tightly distributed and centered away from 0.

#### Predicting new data
I found this one of those things that you look at the slides and follow the steps and nod in agreement from one line to another without really grokking the insight into what's going on. **We can actually get a probability distribution for new data $y_0$ without seeing that data!** As follows:

* We can obtain an expression for $p(y_0 \mid x_0, y, X)$ which is called the **predictive distribution**. We can calculate this exactly.
* This is split into two components: a likelihood and a prior (which is our current posterior!). Note that the likelihood we obtain is the likelihood of $y_0$, not $y$ and we can find expressions for both the likelihood and prior/posterior.
* In other words we get some new data $(y_0, x_0)$ and can calculate $p(y_0 \mid x_0, w)$ which tells us how likely $y_0$ is for a given $w$ and the observed new data $x_0$. We then weight this by our current belief of $w$ which is actually our latest posterior: $p(w \mid y, X)$. This gives us the predictive distribution for the new data point.
* **Note**: we don't actually need $y_0$ for this - this is the point!

#### Active learning: the Bayesian framework allows us to update our model sequentially

Suppose we build a model based on some data $(y, X)$ and estimate our model parameters $w$ from this data i.e. we can calculate the posterior $p(w \mid y, X)$. In the face of new data $(y_0, x_0)$ we are able to write the update step whereby our posterior from $(y, X)$ becomes our new prior.

We thus have something of the form:

New Posterior $\propto$ Likelihood (of new data) * Old Posterior ('the prior')

Or more formally:

$p(w \mid y_0, x_0, y, X) \propto p(y_0 \mid w, x_0) \, p(w \mid y, X)$

The week's project was actually about the situation whereby we have some new data $D = \\{x_1, x_2, ..., x_n\\}$ and a choice about which $y_i$ to measure in order to update our original posterior. For example it might be expensive to measure $y_i$...e.g. digging a hole or conducting an experiment and so the question is whether there is a good way to do this?

The answer to this is shown to be equivalent to choosing the new data $x_0$ that minimizes the entropy of the posterior distribution, or loosely speaking, the data point with the maximum variance as predicted by our predictive distribution.

Here we have: $p(y_0 \mid x_0, y, X) = N(y_0 \mid \mu_0, \sigma_0^2)$ with $\mu_0 = x_0^T \mu$ and $\sigma_0^2 = \sigma^2 + x_0^T \Sigma x_0$.
  * Note: $\mu, \sigma^2$ are the parameters of the posterior $p(w \mid y, X) = N(w \mid \mu, \Sigma)$ which we have already estimated.

#### Sparse regression with the Lasso

Using an $L_1$ (i.e. sum of absolute values) penalty in our objective function performs 'feature selection' and sets some of the parameter weights to 0, thus returning a 'sparse' solution. I'm not going to cover why this is the case but I will say there are many nice geometric explanations if you just type it into your favourite search engine.
  * I am covering this topic less as I am personally quite familiar with the Lasso and so instead will write a little more about some of the mathematical ideas/details that came up in the lectures below.

## Main mathematical ideas from the lectures

* **Bayesian posterior calculation**
  * The continuous version of Bayes rule can be thought of as: $p(\theta \mid data) = \dfrac{p(data \mid \theta) \, p(\theta)}{p(data)}$
    * NB: here I am calling the parameters $\theta$ for simplicity of this discussion.
  * But what is $p(data)$? Recall we have assumed a probabilistic model for our data which is in our numerator as the likelihood multiplied by our prior belief of the parameters. In the numerator we evaluate this likelihood for a given set of parameters e.g. calculate $p(data \mid \theta)$. For the denominator we need to calculate the probability of our data independent of any parameters(!) - this ensures the shape of our posterior from $\theta$ is solely due to the numerator and not the denominator (which is really just a normalising constant).
  * So how do we calculate $p(data)$? Well we can take our numerator and 'integrate out' (i.e. sum/average over all possible values of) any parameters. i.e. $p(data) = \int_{all \, \theta} p(data \mid \theta) \, p(\theta) \, d\theta$
    * I say sum/average because we are multiplying by $p(\theta)$ and so we are weighting each term by a probability so we are really calculating a weighted probability which I guess you can think of as a kind of weighted average but we are also summing over all $\theta$ so it also could be thought of as a sum.
    * **Note 1:**: for multiple parameter systems we use a multiple integral instead of the single I've shown above.
    * **Note 2:** $p(data)$ is usually not calculable (i.e. we cannot compute the integral) though we will see below for the assumptions made in the lecture it has an analytic solution.
  * The lecture's definition of the posterior was: $p(w \mid y, X) = \dfrac{p(y \mid w, X) \, p(w)}{\int_{\mathbb{R}^d} p(y \mid w, X) \, p(w) \, dw}$
    * This is the same formulation as above except we are noting we have to integrate all the parameters out so this means integrating over $\mathbb{R}^d$.
* **Analytic expression for the Bayesian Linear Regression**
  * The lecture showed (after a little bit of work) that we can find an analytic expression for our posterior $$p(w \mid y, X) = N(w \mid \mu, \Sigma)$$ with $$ \Sigma = (\lambda I + \sigma^{-2} X^T X)^{-1}$$ and $$ \mu = (\lambda \sigma^{2} I + X^T X)^{-1} X^T y$$
  * It is noted that $\mu = w_{MAP}$. So Bayesian Linear Regression centres our posterior around the MAP solution except now we have a full description of our parameters, $p(w \mid y, X)$.
* **Extending our notion of a weight penalty using $l_p$ norms**
  * The idea is that the 'norm' of a vector in some way is a measure of its length (hence cannot be negative).
  * The Lasso uses $l_1$ norm penalty introduces sparsity in our solution.
  * The Lasso uses $l_1$ norm penalty which basically means taking the sum of the absolute values of the weight parameter vector $w$. Ridge took the sum of the squares of the elements of $w$ and is referred to as $l_2$ norm. This idea can be extended to take any power, $p$ of as detailed [here](https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm).
  * Depending on how we set $p$ changes how much feature selection we get with the main note that for $p < 1$ we no longer have a convex optimization problem and so need to solve approximately using iterative algorithms.
  * **Note:** For $l_2$ norm as in Ridge the reduction in the cost function we get for reducing any parameter $w_j$ is quadratic whereas for $l_1$ (Lasso) it's linear. And so Ridge penalizes big weights more and will tend to favour all weights of similar magnitude.


## Some mathematical details

Here are some of the mathematical details from the week:

* **Refresher on some of the probabilty rules**:
  * In the predicting new data section we have $p(y_0 \mid x_0, y, X) = \int_{\mathbb{R}^d} p(y_0, w \mid x_0, y, X) \, dw$
    * Here we are simply using the fact that we can integrate a joint distribution of two parameters to obtain the marginal for one of them. The fact we are conditioning on other variables doesn't matter.
  * 'Bring something to the right of the $\mid$ sign', e.g. for $w$:
    * $\int_{\mathbb{R}^d} p(y_0, w \mid x_0, y, X) \, dw = \int_{\mathbb{R}^d} p(y_0 \mid w, x_0, y, X) \, p(w \mid x_0, y, X) \, dw$
    * When we do this we are 'conditioning' on a new variable $w$ and so we must remember to multiply by the probability that this 'event' happens (which may itself be conditioned on other things) e.g. here: $p(w \mid x_0, y, X)$ where we are now conditioning on $w$ which itself depends on $x_0, y, X$.
      * In other words $y_0$ depends on $w, x_0, y, X$ and $w$ depends on $x_0, y, X$.

* **Being able to write matrix multiplication in an additive manner**
  * This is used in the context of active learning to update our posterior sequentially in light of new data. Recall:
    * $p(w \mid y, X) = N(w \mid \mu, \Sigma)$
    * $ \Sigma = (\lambda I + \sigma^{-2} X^T X)^{-1}$
    * $ \mu = (\lambda \sigma^{2} I + X^T X)^{-1} X^T y$
  * Given new data $(y_0, x_0)$ we can calculate:
    * $p(w \mid x_0, y_0, y, X) = N(w \mid \mu, \Sigma)$ by writing:
    * $ \Sigma = (\lambda I + \sigma^{-2} (x_0 x_0^T + \sum_{i=1}^{n} x_i x_i^T))^{-1}$
    * $ \mu = (\lambda \sigma^{2} I + (x_0 x_0^T + \sum_{i=1}^{n} x_i x_i^T))^{-1} (x_0y_0 + \sum_{i=1}^{n} x_iy_i)$
    * **NOTE:** this looks scary but all we have literally done is take old data matrix multiplication of $X^TX$ and now break it out into its components with new data ($x_0 x_0^T$) + old data ($X^TX$)
    * **Q**: why does this work?
      * A: because if we have a data matrix $X$ which is $n$ by $d$ and we wish to calculate $X^TX$ which will be a $d$ by $d$ matrix then we can write this as the sum of the outer products of each data observation $x_i$. Observe that $x_i x_i^T$ returns a $d$ by $d$ matrix. Try it yourself with a simple example!
    * This means as new data comes in we can just perform the calculation for the $x_0 x_0^T$ and $x_0y_0$ and add to our existing matrix sum. In this way we are learning 'sequentially'.
    * Note this will only work if we are adding data to the $n$ dimension as our final matrix dimension doesn't depend on this.
* **Rank one determinant update property of the determinant**
  * This was one of those things just stated, and is basically an unimportant detail as far as conceptual understanding goes. It simply means we can update a matrix multiplication (or even inverse) with new data without having to recalculate the whole thing again. This is similar to what we were doing above with the active learning.
    * You can read more [here](https://dahtah.wordpress.com/2011/11/29/rank-one-updates-for-faster-matrix-inversion/)
* **The least norm solution to the undetermined problem ($d \gg n$)**
  * When we are in the situation with $d \gg n$ we have infinitely many possible solutions. It can be shown by an inequality argument that $w_{ln} = X^T(XX^T)^{-1}y$ has the smallest $l_2$ 'norm' (i.e. length) of all possible solutions.
    * Note: to verify $w_{ln}$ actually solves $Xw = y$ just plug $w_{ln}$ in for $w$ and notice we get $y=y$.
* **Lagrange multipliers**
  * I was actually rusty on this but I can say nothing more to explain them that the legend that is Grant Sanderson can, see the beautiful videos [here](https://www.khanacademy.org/math/multivariable-calculus/applications-of-multivariable-derivatives/lagrange-multipliers-and-constrained-optimization/v/constrained-optimization-introduction).
  * I found the way this was introduced in the lecture a little confusing as it was really brought in just to make a side-point which was another way of verifying that $w_{ln}$ is actually the smallest $l_2$ 'norm' solution of all solutions for $Xw = y$.


## Things I'm unclear on (or outstanding questions)

* TBC

## What did the textbooks say?

To be updated.
