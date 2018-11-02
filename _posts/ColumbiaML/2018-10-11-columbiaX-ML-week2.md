---
layout: post
title: ColumbiaX - ML - week 2
date: 2018-10-11
use_math: true
image: "rr.png"
---

Week 2 of the course covers the link between MLE and OLS, introduces Ridge Regression and Bayes Rule and finishes by showing how RR has a probabilistic interpretation under MAP.

<!--more-->
<hr class="with-margin">

This page is a summary of my notes for the above course, link [here](https://www.edx.org/course/machine-learning-columbiax-csmm-102x-4).

This is my first attempt at publishing course notes and I have no intention to make it comprehensive but rather to highlight the bits I feel are important and maybe explain some of the things I found a little trickier (or weren't explained to my taste). Understanding is deeply personal though if you spot any errors or have any questions please feel free to drop me an email.
<hr class="with-margin">
<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Week 2 (lectures 3 and 4): overview</h4>

Week 2 had a more pleasing finale than week 1 where it felt we really wrapped up some big ideas into a nice summary (though there was no trumpet at the end). Quite a few things were introduced along the way but the main ideas were:

* Showing that under certain assumptions $w_{ML} = w_{LS}$, chiefly that our data is generated from a Gaussian distribution with $y \sim N(Xw, \sigma^2 I)$.
  * Note: this doesn't mean that our target $y$ itself is normally distributed but that **given** a mean for the distribution equal to $Xw$ it (well, the error) follows a Gaussian assumed to have constant variance. We will look at equivalent statements for this below.
* Now we have a probabilistic model for our data which we showed gives us  weights $w_{ML} = w_{LS}$ we can ask further questions about the weights, such as calculating their expected value and variance which is noted can be large under certain conditions.
  * Note: this is a fundamental difference now as we are viewing our weights as random (as they depend on $y$ which we have assumed to be a random variable in our probabilistic model).
* Introduce Ridge Regression (RR) as an alternative to Least Squares (sometimes called OLS, or ordinary least squares) which has more stable solutions.
* Compare RR and LS using the SVD and show how RR is more stable (but *biased*).
* Introduce one of the key ideas in machine learning called the bias-variance trade-off.
* Introduce Bayes rule
* Show how maximizing MAP with a distributional assumption about the weights (along with Bayes rule) gives the same weights as RR. That is (under certain assumptions): $w_{RR} = w_{MAP}$

<hr class="with-margin">
<h4 class="header" id="big">Week 2 (lectures 3 and 4): the big picture</h4>

Again I feel it's easier to state where we will end up as it provides a nice orientation for the details along the way.

##### Loosely speaking we will show ML is to LS what MAP is to RR.

1. **LS and ML:** LS has a probabilistic interpretation if we assume our data is generated as $y \sim N(Xw, \sigma^2 I).$ Maximizing the joint log likelihood of our data in this probabilistic setting is the same as minimizing the LS solution and gives $w_{ML} = w_{LS}$.

2. **RR and MAP:** Ridge Regression adds a penalty term to our objective function. Minimizing this objective function is the same as maximizing the posterior of our parameters given the data (which is proportional to our original data likelihood multiplied by a *prior* on the model parameters, the weights). This further assumption about the distribution of the model's parameters biases the solution we obtain. Maximizing the posterior is called MAP (*maximum a posteriori*) and gives the same solution as Ridge Regression i.e. $w_{MAP} = w_{RR}$.

So ML adds a probabilistic assumption about our data whereas MAP also adds one about the model parameters - we use Bayes rule in MAP.

##### Having a view on the distribution of our model's parameters *biases* them and we see that the bias we introduce trades off with the parameter's variance under a LS setting.

If we solve LS without blindly we are optimising (within the constraints of a linear model) to the particular dataset we have. This can cause us to *overfit* the data and perform worse out of sample, which is what we really care about. Expressing some view (or penalty) about the parameters biases the outcome but may help us be more robust to new data. There are many excellent explanations of the bias-variance trade-off online so I'm not going to repeat them here. The interesting aspect from the lecture was deriving the expected squared error (i.e. the generalization error) of our prediction and showing how this can be written into a form we can recognise as noise, bias and variance.

This problem is the inescapable reality of machine learning although it's usually not possible to get nice equations for the trade-off and so we use cross-validation instead.

<hr class="with-margin">
<h4 class="header" id="math">Main mathematical ideas from the lectures</h4>

* **Deriving $w_{ML}$ from the joint log likelihood, the multivariate Gaussian**
  * In particular it is shown that when $y \sim N(Xw, \sigma^2 I)$ i.e. $\mu = Xw$ then $w_{ML} = w_{LS}.$ This we are in a sense making an independent Gaussian noise assumption about the error $e_i = y_i - x_i^Tw$
  * Note: $w_{ML} = w_{LS} = (X^TX)^{-1}X^Ty$
* **Calculating $E[w_{ML}]$ and $Var[w_{ML}]$**
  * The proof that $E[w_{ML}] = w$ is closely linked to the Gauss-Markov theorem, which isn't mentioned in the lecture but covers the assumptions for the errors $e_i$ which make our parameters unbiased. More [here](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem#Statement).
  * The proof that $Var[w_{ML}] = \sigma^2(X^TX)^{-1}$ is pretty dense and almost certainly not worth focusing on except to note the key thing which is that if $(X^TX)^{-1}$ is large then the variance of our model parameters will be large.
    * When is this the case? Typically when the columns of $X$ are highly correlated.
* **Solving for $w_{RR}$ by differentiating the Ridge Regression objective function similar to what we did for LS**
  * $w_{RR} = (\lambda I + X^TX)^{-1}X^Ty$
  * It is noted that when the regularization parameter $\lambda = 0$ we have $w_{ML} = w_{RR}$
* **Using the SVD to analyse the solutions to LS and RR**
  * This is quite an ugly calculation which really masks the understanding of the result which has nice links back to linear algebra (eigenvalues, PCA etc...). The upshot is that the term $\lambda$ acts as a sort of protection to stop us dividing by really small values when calculating $w_{RR}.$    
    * This is loosely analogous to what we sometimes do to avoid division by 0 errors or blow-ups: i.e. add a term $\lambda$: $1/(\lambda + t)$.
  * In particular it is shown that $w_{RR} = VS_{\lambda}^{-1}U^Ty$ where we are using the SVD to decompose $X$ into 3 separate matrices $X = USV^T$. $S$ is the matrix holding the singular values (i.e. the square roots of the eigenvalues) of $X$. $S_{\lambda}^{-1}$ refers to a diagonal matrix (when it all comes out of the wash by plugging $X = USV^T$ into the definition of $w_{RR}$) with each term of the form: $\dfrac{S_{ii}}{\lambda + S_{ii}}$. $S$ is a $d$ by $d$ matrix.
    * So we finally see (!) how $\lambda$ for $w_{RR}$ stops us getting weights that are huge when we have small singular values in $X$.
* **Calculating the bias-variance trade-off for a general function**
  * Decompose generalization error into noise, squared bias and variance.
* **Probababilistic interpretaton of Ridge Regression using MAP**
  * Likelihood for RR is $y \sim N(Xw, \sigma^2 I)$ as for LS but now if we make a distributional assumption for $w$ as $w \sim N(0, \lambda^{-1}I)$ then we can use Bayes rule to find $w_{MAP}$.
    * We do this by writing $p(w \mid y, X) = \dfrac{p(y \mid w, X) p(w)}{p(y \mid X)}$ which is from Bayes rule.
  * It is shown that under the assumption $w \sim N(0, \lambda^{-1}I)$ that $w_{MAP} = w_{RR}$.

  Note: ML and MAP are what is called 'point estimates', that is, they are the set of parameter values that maximize the posterior - we come onto the distribution of $w$ in the next week.

<hr class="with-margin">
<h4 class="header" id="details">Some mathematical details</h4>

Here are some of the mathematical details from the week:

* Equivalent statements of the assumption we make about our data in a ML setting:
  * $y_i = x_i^Tw + \epsilon_i$ with $\epsilon_i \stackrel{ind}{\sim} N(0, \sigma^2)$ for $i=1,...n$
  * $y_i \stackrel{ind}{\sim} N(x_i^Tw, \sigma^2)$ for $i=1,...n$
  * $y \sim N(Xw, \sigma^2 I)$
* Probability fact: if $y \sim N(\mu, \Sigma)$ then: $E[yy^T] = \Sigma + \mu\mu^T$
  * Note this is actually just from $\Sigma = Cov(X,Y) = E[XY] - E[X]E[Y]$
* Note for any real matrix $X^TX$ is symmetric
* How to take the derivative of $L = (y-Xw)^T(y-Xw) + \lambda w^T w$ with respect to $w$, i.e. compute $\nabla_w L$.
* SVD knowledge: writing any $n$ by $d$ matrix $X$ as $X = USV^T$.
  * $U$ is $n$ by $d$ with orthonormal columns
  * $S$ is $d$ by $d$ with non-negative diagonal entries.
  * $V$ is $d$ by $d$ with orthonormal columns
* It was stated in the lecture that our squared prediction error for a new data point $(x_0, y_0)$ is: $$E[(y_0 - x_0^T\hat{w})^2 \mid X, x_0] = \int_{\mathbb{R}}^{} \int_{\mathbb{R^n}}^{} (y_0 - x_0^T\hat{w})^2 \, p(y \mid X, w) \, p(y_0\mid x_0, w) \, dy \, dy_0$$
  * First observation: WTF
  * Second observation: WTF I hate big integrals
  * Third observation: I'll come back to this
  * Fourth observation: Oh hang on, this just looks like a use of the fact that $E[X \mid A] = \int_{x} x \, p_{X \mid A}(x) \, dx$
  * The key with this formula is to remember a few that when you see $E[X \mid A]$ the bit to the left of $\mid$ you can literally just stick into the integral. In this case this is: $(y_0 - x_0^T\hat{w})^2$.
  * The probability distribution bit then is the densities of the random parts of the equation remembering to condition on anything we need to. In this case the probability distribution  bit has been broken down into (I think):
    * $p(y_0 \mid X, x_0, w) = p(y \mid X, w) \, p(y_0\mid x_0, w)$
  * We then integrate (sum) over all our predictions noting that $y_0$ is a single observation and $y$ is $n$ observations.
* Note that in $E[(y_0 - x_0^T\hat{w})^2 \mid X, x_0]$ both $\hat{w}$ and $y_0$ are treated as random as $\hat{w}$ is found using $y$ (which is a RV) and then is used to predict $y_0$.

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

* Why does: $p(y_0 \mid X, x_0, w) = p(y \mid X, w) \, p(y_0\mid x_0, w)$ ?
  * Here $(y, X)$ represents original data and $(y_0, x_0)$ new data we receive.
* Why under the model $y \sim N(Xw, \sigma^2 I)$ when we calculate $E[w_{ML}]$ do we have: $E[w_{ML}] = \int [(X^TX)^{-1}X^Ty]\,p(y \mid X, w)\, dy$ ?
  * Recall: $w_{ML} = (X^TX)^{-1}X^Ty$

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>

To be updated.
