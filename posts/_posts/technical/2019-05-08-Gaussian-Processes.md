---
layout: post
title: Gaussian processes - a first encounter
date: 2019-05-08
use_math: true
tags: ['gaussian_processes', 'probabilistic_modelling', 'kernels']
image: "gp.png"
comments: true
---
In a loose sense Gaussian processes (GPs) can be thought of as a probabilistic yet non-parametric form of non-linear regression which sit within the Bayesian framework. In this article we give an in-depth introduction to the topic collating many of the excellent references on the topic.

<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="notation">Notation summary</h4>
This post combines many sources and so we will try the best we can to be consistent in notation.

In general, capital letters are matrices, bold font represents vectors and lower-case letters are scalars.

* $X$: $n \times d$ data matrix with each row an observation
* $X_{+}$: test data matrix we wish to predict the target variable for, $n_{+} \times d$
* $\mathbf{x}\_{i}$: $i$th observation of data with $d$ elements
* $\mathbf{x}\_{+}$: single test point with $d$ elements
* $\mathbf{y}$: target variable of length $n$
* $\mathbf{y}\_{+}$: test values of target for new data, length $n_{+}$, generally unobserved
* $y_i$: target value for $i$th observation
* $\mathcal{D}$: some data containing both features and target variable, i.e. $\\{X, \mathbf{y}\\}$
* $\mathbf{f}$: vector representing the Gaussian process mean for each data point, $\mathbf{f} = (f(\mathbf{x}\_{1}), ..., f(\mathbf{x}\_{n}))$
* $\mathbf{f}\_{+}$: predictions for the target variable for new data, $X_{+}$
* $f_i$: shorthand for $f(\mathbf{x}_i)$, the function evaluated for the $i$th observation
* $f_{+}$: shorthand for the prediction for a single test point, $\mathbf{x}\_{+}$
* $\kappa(\mathbf{x}\_{1}, \mathbf{x}\_{2})$: kernel function evaluated at two points - returns a scalar
* $\kappa(A, B)$: kernel function evaluated for two matrices $A$ and $B$ - returns a matrix with dimensions $m \times p$ for $A$ with dimensions $m \times d$ and $B$ with dimensions $p \times d$
* $K$: $\kappa(X, X)$, a $n \times n$ matrix, where the $i , j$ th entry is $\kappa(\mathbf{x}\_{i}, \mathbf{x}\_{j})$
* $K_{+}$: $\kappa(X, X_{+})$, a $n \times n_{+}$ matrix
* $K_{\++}$: $\kappa(X_{+}, X_{+})$, a $n_{+} \times n_{+}$ matrix
* $m(\mathbf{x}\_{i})$ or $m(X)$: $m$ is some function that models the mean of an observation, often set to be $m(\mathbf{x}\_{i}) = 0$ for $i = 1, ..., n$
* $\boldsymbol{\mu}$ or $\mu(X)$: is $(m(\mathbf{x}\_{1}), m(\mathbf{x}\_{2}), ..., m(\mathbf{x}\_{n}))$ a vector of length $n$
* $\mu_{+}$: mean for a single test point
* $\mu_{f_{+} \| \mathbf{f}}$: mean vector for predicted data after conditioning on observed data
* $\Sigma_{f_{+} \| \mathbf{f}}$: covariance matrix for predicted data after conditioning on observed data

<hr class="with-margin">
<h4 class="header" id="outline">Outline</h4>

Here is a roadmap of what is covered in this post:

* Introduction
  * Kernels and functions as vectors
  * Visualizing multivariate Gaussians
  * Posterior predictive view
  * Define a GP
* GP regression (noise free)
  * Derive predictive equations
  * Plot example
  * Code
* GP regression (noisy)
  * Derive predictive equations
  * Plot example
  * Code    
* Recipe for a GP
* Take home messages
* Key mathematical ideas and results
* Q and A
  * How to get kernel parameters
* References

<hr class="with-margin">
<h4 class="header" id="intro_gp">Gaussian process preliminaries</h4>

<blockquote class="tip">
<strong>TLDR:</strong> GPs give a way to express a view on functions. In particular they allow us to specify how smooth we expect a function to be rather than how many parameters we expect it to have. The key idea they rely on is that data points that are close in input space are expected to be similar in output space, i.e. if $\mathbf{x}_{i}$ and $\mathbf{x}_{j}$ are similar then $f(\mathbf{x}_{i})$ and $f(\mathbf{x}_{j})$ will be close in value too.
</blockquote>

##### Introduction

Gaussian processes are Bayesian alternatives to kernel methods and allow us to infer a _distribution over functions_, which sounds a little crazy but is actually both an intuitive thing to desire as well as being analytically tractable in certain cases.

Typically in machine learning we have some features $X$ with labels $\mathbf{y}$ and we assume that $\mathbf{y} = f(X)$ for some function $f$. When making this assumption we are typically fixing the parameterisation capacity of the model we wish to use, for example, in linear regression we are assuming there are slope and intercept terms and so the total amount of parameters are fixed given the data.

What if, instead, we wished to think about all possible functional forms for $f$ without pre-specifying how many parameters are involved? It turns this is what Gaussian Processes allow us to do and in return we must specify a prior over the type of functions we wish to see.

##### Kernels and functions as vectors

It is worth briefly explaining a few concepts.

###### Kernels

In the context of Gaussian processes a kernel function is a function that takes in two vectors (observations in $\mathbb{R}^ d$) and outputs a similarity scalar between them. As covariance matrices must be positive semi-definite valid kernels are those that satisfy this requirement. For GPs we evaluate the kernel function for each pairwise combination of data-points to retrieve the covariance matrix. The covariance matrix will end up not only describing the shape of our learned distribution, but ultimately determines the characteristics of the function that we want to predict.

We can go further and say that the problem of learning with Gaussian processes is exactly the problem of learning the hyper-parameters of the kernel function - this will be discussed in greater detail later.

There are many kernels that can describe different classes of functions, including to encode properties such as periodicity. In this post we will restrict ourselves to the most common kernel, the [radial basis function (or Gaussian) kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel):

$$
\kappa(\mathbf{x_i}, \mathbf{x_j}) = \sigma^{2} \exp (-\frac{ \| \mathbf{x_i} - \mathbf{x_j} \|^{2}}{2 l^{2}})
$$

with hyper-parameters $\sigma$ and $l$. The variance $\sigma^2$ determines the magnitude of fluctuation of values away from the mean and $l^2$ determines the reach of neighbouring data-points (small $l$ will give wiggly functions, increasing $l$ gives smoother functions). See the plot below for the effect of different values of the hyper-parameters.

<p align="center">
    <img src="/assets/img/gp_kernel_params.png" alt="Image" width="600" height="400" />
</p>
<em class="figure">Effect of different kernel hyper-parameters</em>

Kernel functions are also sometimes referred to as covariance functions.

<blockquote class="tip">
<strong>Caution:</strong> Gaussian processes get their name because they define a Gaussian distribution over a vector of function values, not because they sometimes use a Gaussian kernel to determine the covariances.
</blockquote>

###### Functions as vectors
Loosely speaking a function can be viewed as an infinitely long vector. We could imagine discretizing the input space with a huge grid of values containing every possible combination of floating point numbers for each dimension of our matrix $X$. Theoretically (but not practically) we could then evaluate $\mathbf{f} = (f(\mathbf{x}\_{1}), ..., f(\mathbf{x}\_{n}))$ for this huge $n$ and so $\mathbf{f}$ would be a very large vector now containing the function's values for the domain of interest.

Alas, we can’t store a vector containing the function values for every possible input (recall each $\mathbf{x_i}$ is $d$ dimensional), though it is possible to define a multivariate Gaussian prior on it. Given (potentially) noisy observations of some of the elements of this vector, it will turn out that we can infer other elements of the vector without explicitly having to represent the whole object.

The miraculous thing about GPs is that they provide a rigorous, elegant and tractable way to deal with the above problem.

Note that whilst talk of functions as vectors can seem a little hand-wavy it can be made rigorous.

##### A different way to view multivariate distributions

In order to aid the discussion on large dimensional Gaussians it's crucial to switch to a different way to visualize them. We start by thinking of each data-point $\mathbf{x_i}$ having $d$-dimensions, so $X$ is $n \times d$ and $\mathbf{x_1}$ and $\mathbf{x_2}$ refer to 2 observations from our data where $\mathbf{y} = \\{y_1, y_2\\}$ is the function value for each data-point.

Using the kernel function we can calculate the covariance matrix, $\Sigma$, between any 2 points (using $X$ only), for example:

$$
\Sigma=\left[ \begin{array}{ll}{1} & {.7} \\ {.7} & {1}\end{array}\right]
$$

and by the assumption we make for GPs (see [below](#gp_defn)) we assume the output function values are jointly distributed according to a Gaussian:

$$
p(\mathbf{y} | \Sigma) \propto \exp \left(-\frac{1}{2} \mathbf{y}^{\top} \Sigma^{-1} \mathbf{y}\right)
$$

Using this idea it is then possible to visualize what the 2-dimensional Gaussian looks like and draw samples from it. Notice that the covariance between function values is fully specified by the corresponding input values, and not in any way by the actual values of the function; this is a property of the Gaussian Process.

The left hand plots in the below image show contour/density plots for a 2-dimensional Gaussian with a given correlation coefficient (in reality this is calculated by the kernel function) and zero mean vector. The red points shown in the left hand plots are samples from this 2-dimensional Gaussian.

<p align="center">
    <img src="/assets/img/gp_contours.png" alt="Image" width="650" height="500" />
</p>
<em class="figure">A different way to visualize a sample from a 2-dimensional Gaussian</em>

The right hand plots show an alternate way to visualize this sample, where the y-axis represents the value of $y_1$ and $y_2$ respectively. In this way we could easily imagine calculating the covariance matrix between more data-points and plotting a sample in the way shown in the right hand plots, but the left hand plots don't allow this view. This is how we are able to represent multi-dimensional Gaussians on a single plot, as below:

<p align="center">
    <img src="/assets/img/gp_many_dims_single.png" alt="Image" width="500" height="350" />
</p>
<em class="figure">A single sample from a multi-dimensional Gaussian</em>


<a name="prelim_plot"></a>
###### Hold on, what's the x-axis for the above plot?

In the above plot we ordered the points according to their index in the covariance matrix but this was simply for illustrative purposes only. In general, the x-axis will be the values of $X$ for which we wish to calculate a point at and thus can take on any real-value. Note once $d > 1$ we can't think of the x-axis as being 1-dimensional any more and so even the above plot breaks down. The input space is simply a  grid (arbitrarily fine if we wish) for which we can calculate function values at.

Nevertheless, when $d=1$ this way of visualizing helps us plot multi-dimensional Gaussians a lot easier and will be helpful for the examples we encounter in the rest of this post.

<blockquote class="tip">
<strong>Remember:</strong> for GPs the multi-dimensional aspect of Gaussians refers to the number of data-points, not the dimensionality, $d$, of $X$. Once we have computed the covariance matrix via the kernel function we can essentially throw the original $X$ data away.
</blockquote>

##### Probability perspective
Before we start with the equations for GPs we draw a link to traditional modelling in machine learning. Typically in machine learning the Bayesian approach to modelling involves inferring the posterior distribution of the parameters of some model given data, i.e. $p(\theta \| \mathcal{D})$. GPs allow us instead to model $p(f \| \mathcal{D})$ and we can then use this modelled function to predict on new data, $X_{+}$:

<div class="math">
\begin{align*}

\underbrace{p\left(f_{+} | X_{+}, X, \mathbf{y}\right)}_\text{posterior predictive} &= \int \underbrace{p\left(f_{+} | f, X_{+}\right)}_\text{likelihood} \, \underbrace{p(f | X, \mathbf{y})}_\text{posterior} \, df \\[5pt]
&= \int p\left(f_{+}, f | X, X_{+}, \mathbf{y}\right) \, df

\end{align*}
</div>

and so for each new test point in $X_{+}$ we obtain a probability distribution for the corresponding prediction in $f_{+}$. It turns out for the regression case that the above can be done in closed form using known statistical results and linear algebra.

Whilst the above formulation is useful in linking Gaussian processes into the wider Bayesian framework (by showing a likelihood and a prior) it's not typically the most intuitive place to start. For that, it's much easier to start by defining a Gaussian process and rely upon some key properties of the multivariate Gaussian distribution. These are given [below](#mult_rvs) and assumed from now on - though we will make reference to them as we use them.

<a name="gp_defn"></a>
##### Definition of a Gaussian process
The derivation of the predictive update equations for GPs follow from the definition and assumptions made within it, and this definition is where we will start.

<blockquote class="tip">
<strong>Definition</strong>
<br>
A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.
<br>
<br>
<strong>Note:</strong> a Gaussian process is completely specified by its mean function and covariance function.
</blockquote>

Thinking of a GP as a Gaussian distribution with an infinitely long mean vector and an infinite by infinite covariance matrix may seem impractical but luckily we are saved by the [marginalization property](#mult_rvs) of Gaussians. That is, the whole thing is tractable as long as we only ever ask finite dimensional questions about functions - these finite points are where we have data or wish to predict.

It is also worth clarifying that in the case of Gaussian processes each data point is a random variable and thus the multivariate Gaussian has the same number of dimensions as the number of data-points: for the training data it is a $n$-dimensional Gaussian. Once we add test data for which to predict the Gaussian will have $n + n_{+}$ dimensions.

We now move onto the predictive equations for the noise-free case.

<hr class="with-margin">
<h4 class="header" id="gp_reg_no_noise">GP regression (noise-free)</h4>

We will walk through a small example with a small amount of data, the below holds for data of any size.

##### Problem set-up
We start by assuming we are given 3 training data-points $\mathcal{D} = \\{ (x_1, f_1), (x_2, f_2), (x_3, f_3)\\}$ and a new test point $x_{+}$ for which we wish to predict $f_{+}$. We assume that the training data has no noise and contains samples from the true unknown function, that is, $f_i = y_i$ for all $i$.

By the definition of a GP it is assumed that any set of random variables, which are the output function values for the data-points, are distributed as a multivariate Gaussian. Given this assumption we can write the joint distribution between our observed training data and the predicted output as:

$$
\left[ \begin{array}{l}{\mathbf{f}} \\ {f_{+}}\end{array}\right]
\sim
\mathcal{N}
\Bigg(
\left[ \begin{array}{c}{\boldsymbol{\mu}} \\ {\mu_{+}}\end{array}\right] \, , \,

\left[ \begin{array}{cc}{K} & {K_{+}} \\ {K_{+}^T} & {K_{\\++}}\end{array}\right]
\Bigg)
$$

It is normally common to assume the mean functions are 0 for Gaussian processes, the reasons for why this is are discussed in the Q and A section [below](#mean_modelling).

<blockquote class="tip">
<strong>Sidebar on the covariance sub-matrices</strong>
<br>

The above sub-matrix $K$ contains the similarity of each training point to every other training point:

$$
K =
\left[
\begin{array}{cc}
{K_{11}} & {K_{12}} & {K_{13}} \\
{K_{21}} & {K_{22}} & {K_{23}} \\
{K_{31}} & {K_{32}} & {K_{33}}
\end{array}
\right]
$$

where each entry $K_{ij} = \kappa(x_i, x_j)$ is the evaluation of the kernel function between any two data-points.

Similarly $K_{+}$ is a $3 \times 1$ vector which contains the evaluation of the similarity between each training data-point and the test point $x_{+}$. $K_{\\++}$ in this case is a scalar with the similarity between the test point to itself.
<br>
<br>
The above generalizes to many data points and in general $K$ has dimensions $n \times n$, $K_{+}$ has dimensions $n \times n_{+}$ and $K_{\\++}$ has dimensions $n_{+} \times n_{+}$.
</blockquote>

##### Condition then marginalize

Using the properties of multivariate Gaussians mentioned [above](#mult_rvs) we are able to quickly write down the predictive equations for GPs. Note that in contrast to other Bayesian analysis we do not need to trouble ourselves with computing a posterior explicitly, nor do we need to explicitly perform any integration.

Instead we condition on the observed training data, $f$, to obtain the posterior predictive (conditional) distribution for $f_{+}$. Using the result for the [conditioning](#mult_rvs) of Gaussians we have, for a single test point:

<div class="math">
\begin{align*}

p(f_{+} | X_{+}, X, \mathbf{f}) &= \mathcal{N}\left(f_{+} | \mu_{+}, \Sigma_{+}\right) \\[5pt]
\underbrace{\mu_{f_{+} | \mathbf{f}}}_\text{predictive mean} &= \underbrace{\mu\left(x_{+}\right) + K_{+}^{T} K^{-1}(\mathbf{f} - \mu(X))}_\text{linear in $\mathbf{f}$} \\[5pt]
\underbrace{\Sigma_{f_{+} | \mathbf{f}}}_\text{predictive uncertainty} &= \underbrace{K_{\\++}}_\text{prior uncertainty} - \underbrace{K_{+}^{T} K^{-1} K_{+}}_\text{reduction in uncertainty}
\end{align*}
</div>

The above conditioning reduces a 4-dimensional Gaussian down to a 1-dimensional Gaussian - we can thus think of the conditioning as cutting 3 slices in the 4-dimensional space to leave us with a 1-dimensional distribution. In general, once we have conditioned on the observed data we will be left with a multivariate Gaussian with dimensionality equal to the number of test points.

To then obtain the marginal distribution of each test point we use the marginalization property of multivariate Gaussians to obtain the mean prediction and variance estimate for each point (in this case there is no need to as we are left with a 1-dimensional distribution for our single test point). In this way the prediction is not just an estimate for that point, but also has uncertainty information - for understanding why knowing uncertainty is important see the Q and A [below](#know_uncertainty).

##### Adding more data

In the above we have 3 observed data points and made a prediction for a new test point. If we wish to predict for more test data we can simply repeat the above steps and thus we are able to get probabilistic predictions for any new test point. In a similar spirit if we observe new training data we can add this in and recalculate the predictions for the test points. (Note this ignores computational considerations to keep things simple.)

##### Walking through an example with plots
We now give an example to illustrate the above discussion.

In this example we are given 3 training data-points $\mathcal{D} = \\{(-2, 1), (1, -1.5), (4, 2)\\}$ and a new test point $x_{+} = 3$ and we wish to predict $f_{+} = f(3)$.

The below chart visualizes parts of the GP fit process and is explained below:

<p align="center">
    <img src="/assets/img/gp_blog_plots.png" alt="Image" width="700" height="500" />
</p>
<em class="figure">Gaussian process example plots</em>
<hr class="with-margin">

###### Subplot 1: just the data
Without a model and just visual inspecting the 3 training data-points we might guess that $f_{+} \approx 0$ and we would also perhaps expect it to lie in the range around this point. This is an informal way of thinking about the key idea behind GPs - that we expect points that are close in input space to be close in output space also.

###### Subplot 2: sampling from the prior
By defining the kernel function, and before observing any training data, we are specifying a view on how smooth we expect the estimated functions to be. Sampling from the prior amounts to having to specify some domain of interest (here we use -5 to 5) and discretizing it into many points. This is what we loosely mean by saying a function is just a vector of values. This subplot essentially shows a 50-dimensional Gaussian as this is the number of points used along the x-axis - note this creates realistic plots and is a key point. We rarely ever care about every single value our input data can take but just a subset where we have training data or wish to predict.

The data for this subplot is not part of the training data and we just show these samples for illustration. These functions will become constrained once we observe some data.

###### Subplot 3: fit the Gaussian process
Here we have fit the GP regression model and show the outcome. The blue dot is the mean value for the GP prediction and the green dashed line show $\pm 2$ standard deviations around this predicted mean value.

###### Subplot 4: sampling from the posterior
Similar to what we did when we sampled from the prior we can predict the GP for a range of x values, here do it for many points along the x-axis and show the resulting mean vector (red-dashed line) and $\pm 2$ standard deviation estimates for each prediction. Note that where we have observed data the functions from the posterior are constrained to go through the data but in the places where we have no data the possible function fits are many and varied.

##### Comment
Recall that we can represent a function as a big vector where we assume this unknown vector was drawn from a big correlated Gaussian distribution, this is a Gaussian process. As we observe elements of the vector this constraints the values the function can take and by conditioning we are able to create a posterior  distribution for functions. This posterior is also Gaussian, i.e. the posterior over functions is still a Gaussian process. The marginalisation property of Gaussian distributions allows us to only work with the finite set of function instantiations $\mathbf{f} = (f(\mathbf{x}\_{1}), ..., f(\mathbf{x}\_{n}))$ which  constitute the observed data and jointly follow a marginal Gaussian distribution.

<hr class="with-margin">
<h4 class="header" id="post_first">A different approach - posterior first</h4>

We present a slightly different approach to GPs by reasoning from Baye's rule in order to obtain the predictive equations. For the below analysis it will pay to bear in mind Baye's rule in the form below (the extra conditioning variable $X$ just gets carried along for the ride):

$$p(f|X, y) = \dfrac{p (f, y | X)}{p(y|X)} = \dfrac{p(y | X, f) \, p(f | X)}{p(y|X)} $$

We start by assuming the prior, $p(f \| X)$, is a Gaussian process, that is:

$$ p(f | X) = \mathcal{N}(m(X), \kappa(X, X)) $$

or

$$ f(X) \sim \mathcal{N}(m(X), \kappa(X, X)) $$

where $m(X)$ is a function that models the mean and $\kappa(X, X)$ a function that models the covariance matrix - we will come back to these shortly.

What about the likelihood, $p(y \| X, f)$? In the case of additive noise we assume the data has the form, for every observation:

$$y = f(x)+\epsilon$$

where $\epsilon$ is an unknown noise variable, assumed Gaussian and independent on each data point, i.e. $\epsilon \sim \mathcal{N}\left(0, \sigma_{e}^{2}\right)$.

Under these assumptions the likelihood is also Gaussian:

$$
p(y | X, f)=\prod_{i=1}^{N} \mathcal{N}\left(y_{i} ; f_{i}, \sigma_{\epsilon}^{2}\right)=\mathcal{N}\left(y ; f, \sigma_{\epsilon}^{2} \, I_{N}\right)
$$

In other words, both the prior and likelihood are Gaussian distributions which means the posterior will also be a Gaussian with a known closed form.

To combine the prior and the likelihood is to multiply two Gaussian density functions.

<div class="math">
\begin{align*}

p(f, y | X) &= p(y | X, f) \, p(f | X) \\[5pt]
&= \mathcal{N}\left(y ; f, \sigma_{\epsilon}^{2} \, I_{N}\right) \,  \mathcal{N}(m(X), \kappa(X, X))
\end{align*}
</div>

which leads to (see box below), after the application of standard results, the following expression for the joint density:

$$
\left[ \begin{array}{l}{f} \\ {y}\end{array}\right]
\sim
\mathcal{N}
\Bigg(
\left[ \begin{array}{c}{m(X)} \\ {m(X)}\end{array}\right] \, , \,

\left[ \begin{array}{cc}{K} & {K} \\ {K} & {K+\sigma_{\epsilon}^{2} I_{N}}\end{array}\right]
\Bigg)
$$

<blockquote class="tip">
<strong>Sidebar on the joint distribution</strong>
<br/>
Explanation of the above expression.

The mean of the joint distribution relies on the fact that the resulting mean, $\mu$ of the product of two Gaussian densities with the same means (i.e. $\mu_1 = \mu_2$) is $\mu = \mu_1 = \mu_2$ from:

<div class="math">
\begin{align*}
\mu &= \frac{\sigma_{1}^{-2} \mu_{1}+\sigma_{2}^{-2} \mu_{2}}{\sigma_{1}^{-2}+\sigma_{2}^{-2}} \\[5pt]
\end{align*}
</div>

this is regardless of the values of $\sigma_{1}$ and $\sigma_{2}$ (factor the above).

The covariance matrix is a little trickier to explain but start by recalling both $f$ and $y$ are being evaluated for the same data, $X$. The covariance matrix of the joint distribution is:

<div class="math">
\begin{align*}
\Sigma &= \left[ \begin{array}{cc}{\Sigma_{ff}} & {\Sigma_{fy}} \\ {\Sigma_{yf}} & {\Sigma_{yy}}\end{array}\right] \\[5pt]
&= \left[ \begin{array}{cc}{K(X, X)} & {K(X, X)} \\ {K(X, X)} & {K(X, X) + \sigma_{\epsilon}^{2} I_{N}}\end{array}\right] \\[5pt]
\end{align*}
</div>

where the above result stems from the fact that marginal of a joint Gaussian is also Gaussian. The noise term appears in $\Sigma_{yy}$ only due to the assumption about the independence of the noise and that it is only $y$ that is a noise added process.

More reading on the technical details of the above can be found in the reference section at the end of the blog.

</blockquote>

We can now apply standard Gaussian conditioning to obtain an expression for the posterior:

$$
p(f | y, X)=\mathcal{N}\left(m(X)+K\left[K+\sigma_{\epsilon}^{2} I_{N}\right]^{-1}(y-m(X)), \quad K-K\left[K+\sigma_{\epsilon}^{2} I_{N}\right]^{-1} K\right)
$$

For the relevant theory of how to derive the above, refer to [Murphy](https://www.cs.ubc.ca/~murphyk/MLbook/), section 4.3.1.

###### Looking at the posterior predictive

Most discussions on GPs usually start with the statement of the joint and THEN conditioning BLAH posterior predictive and show conditioing leads

###### What are $m(X)$ and $\kappa(X,X)$?
TODO: This has different notation atm

where we have that $m(x)$ is the mean function and $k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)$ is the covariance function. Requiring that

$$
\begin{aligned} m(\mathbf{x}) &=\mathbb{E}[f(\mathbf{x})] \\ \kappa\left(\mathbf{x}, \mathbf{x}^{\prime}\right) &=\mathbb{E}\left[(f(\mathbf{x})-m(\mathbf{x}))\left(f\left(\mathbf{x}^{\prime}\right)-m\left(\mathbf{x}^{\prime}\right)\right)^{T}\right] \end{aligned}
$$

$$
f(\mathbf{x}) \sim \mathcal{G P}\left(m(\mathbf{x}), k\left(\mathbf{x}, \mathbf{x}^{\prime}\right)\right)
$$


###### A function $f$ is a random variable?


###### Tying up the dimensions



##### Grokking this shizzle

Notice that the covariance between function values is fully specified by the corresponding input locations, and not at all by the actual values of the function; this is a property of the Gaussian Process.




<hr class="with-margin">
<h4 class="header" id="small_example">Recipe for a Gaussian process</h4>
We will now walk through a small example detailing both the maths and the code:

<hr class="with-margin">
Recipe for GP example (before we have observed any data):
* Define the kernel function $\textbf{k}$ which will output matrices we call $K$, $K_s$ or $K_{ss}$ depending on which data points we are computing similarity between (e.g. train to train, train to test, test to test).
* Pick the domain we are interested in $X_{test}$, here $-5 \leq x \leq 5$ and compute the kernel between all evenly spaced points in this region.
  * Note as $X_{test}$ is evenly spaced the kernel will only have strong values down the diagonal as we are saying further points are less related.
* Sample from $f_s \sim N(\mu_s, \Sigma_s)$ to see what the functions look like. To actually sample we actually use $f_s \sim \mu_{s} + L \, N(0, I)$ with $LL^T = K_{ss}$ which involves using a Cholesky decomposion on $K_{ss}$.
  * Note this is really just using the fact that $x \sim N(\mu, \sigma^2)$ can be written as $x \sim \mu + \sigma N(0, 1)$ except we have a matrix instead of $\sigma$ and so taking the 'square root of a matrix' is what the Cholesky decomposition does, loosely.
  * We assume in this example that $\mu_s$ = 0.

<hr class="with-margin">

Example python code of the small GP example:
<pre><code class="language-python">import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b):
    """ GP squared exponential kernel i.e. the distance/similarity function"""
    kernelParameter = 0.1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

n = 50  # number of test points
Xtest = np.linspace(-5, 5, n).reshape(-1, 1)  # test points
K_ss = kernel(Xtest, Xtest)  # kernel at test points
# Draw samples from the prior at the test points:
L_prior = np.linalg.cholesky(K_ss + 1e-6*np.eye(n))   # K = LL^T and L is (n, n)
f_prior = np.dot(L_prior, np.random.normal(size=(n, 3)))  # Sample 3 points per f_s ~ LN(0, I)
plt.plot(Xtest, f_prior) # these are now samples from the GP prior
</code></pre>


Plotting this we get a look at the prior $p(f_s)$:

<p align="center">
    <img src="/assets/img/gprior_simple.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 1: Gaussian Prior samples</em>
<hr class="with-margin">

##### Comments on what we just did

* We say we are 'sampling functions' as we could technically compute each sample at more and more points in the $x$ domain instead of the 50 we did here.
* All the functions are different but they all come from the same distribution whose covariance (dictated by the choice of kernel) controls the smoothness of these functions (i.e. how far apart is the function value for two input points that are close by allowed to be).
* We call this the prior even though it's over what we termed the $X_{test}$ domain as it's the domain of interest which we wish to predict for. We thus view the sample of functions over this domain before seeing any data to get a sense of what they look like.
* When we then see some data this will reduce the set of functions from $p(f_s)$ to the posterior $p(f_s \mid f) = p(f_s \mid X_{test}, X, y)$ .

<hr class="with-margin">
<h4 class="header" id="extended_example">Extended GP example</h4>

Let's see some data now and see what happens.


<hr class="with-margin">
Recipe for GP example (after observing data):

* As for the smaller example we can calculate the prior and sample from it
* We now create some training data from an underlying function and add some noise to it (as we are pretending we observe a noisy version of it)
* We can then calculate the relevant values in order to obtain the posterior distribution which we can sample from. In particular we have that:

$$p(f_s \mid X_s, X, y) = N(f_s \mid \mu_s, \Sigma_s)$$

$$\mu_s = K_s^T K_y^{-1}y$$

$$\Sigma_s = K_{ss} - K_s^T K_y^{-1}K_s$$

where we are calling $K_y = K + noise$ (i.e. the kernel of the noisy data we see).

<hr class="with-margin">

Example python code of the extended GP example (excluding plotting code):

<pre><code class="language-python">import numpy as np
# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()

# Inputs
N = 10         # number of observed training points
n = 50         # number of test points to predict for
s = 0.0005     # noise variance to create data with

# 1. As before create data for domain we're going to make predictions at (i.e. evenly spaced) and sample:
Xtest = np.linspace(-5, 5, n).reshape(-1,1)
K_ss = kernel(Xtest, Xtest)
L_prior = np.linalg.cholesky(K_ss + 1e-6*np.eye(n))
f_prior = np.dot(L_prior, np.random.normal(size=(n, 3)))

# 2. Observe some data get the noisy versions of the function evaluated at these points (this is the training data)
X = np.random.uniform(-5, 5, size=(N, 1))  # (10, 1)
y = f(X) + s*np.random.randn(N) # (10, 1)
K = kernel(X, X)  # N by N matrix of distances for training data
L = np.linalg.cholesky(K + s*np.eye(N))  # K = LL^T

# 3. We also need to compute the distance between the train and test data
alpha = np.linalg.solve(np.dot(L, L.T), y)
K_s = kernel(X, Xtest)

# 4. Get the mean and variance over whole test domain
mu = np.dot(K_s.T, alpha)
Lk = np.linalg.solve(L, K_s)
s_var = np.diag(K_ss) - np.sum(Lk**2, axis=0)
stdev = np.sqrt(s_var)

# 5. If we want, draw samples from the posterior within test domain
L_post = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L_post, np.random.normal(size=(n, 3)))
</code></pre>

Plotting this we get the following:

<p align="center">
    <img src="/assets/img/predict.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 2: True function (blue), fitted function (red dash), noisy observed data (red cross) and 3*standard deviation (grey)</em>

<p align="center">
    <img src="/assets/img/post.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 3: Gaussian Posterior samples</em>
<hr class="with-margin">

##### A few comments on the above charts

* The uncertainty bands are wide when we have no observed data. In fact they are so wide even when we have data and actually look a bit silly. This is because the prior we specified was very flexible i.e. we chose a kernel that allowed functions which are not very smooth (relative to the underlying data generating process, a sine wave). We can change the kernel parameter to fix this, see figure 4 below.

<p align="center">
    <img src="/assets/img/predict_smooth.png" alt="Image" width="350" height="250" />
</p>

<em class="figure">Fig. 4: Gaussian Posterior with smoother kernel parameter (different data)</em>

* This looks much better as we are now more confident in regions where we have seen more data. Note that even if we specified a poor prior as we kept seeing more data it would eventually overwhelm the poor assumption. That said, GPs are powerful when we are able to make strong prior choices about the data generating process and thus can perform well with data paucity if we get this assumption correct.
* In figure 3 above when we sampled from the posterior what we are essentially doing is now drawing samples from the updated distribution in light of the data we have observed. As we can the samples now look a little closer the data we observed.


<hr class="with-margin">
<h4 class="header" id="take_home">Take-home messages</h4>


[Summary](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf)
Gaussian Process Take-Home Message

Gaussian processes are non-parametric.

A Gaussian process is a collection of random variables,any finite number of which have joint Gaussian distributions.

A Gaussian process is fully specified by a mean function and a covariance function.

The problem of learning with Gaussian processes is exactly the problem of learning the hyperparameters of the covariance function.

Basic rules of multivariate Gaussian distributions govern manipulation of the Gaussian process after a finite number of data points is observed.


<hr class="with-margin">
<h4 class="header" id="math">Key mathematical results and ideas</h4>
<a name="mult_rvs"></a>
##### Multivariate Gaussians - 2 key results

We do not give an exhaustive overview of the many properties of multivariate Gaussians but instead give two key properties which are of paramount importance for the use of Gaussian processes.

<blockquote class="tip">
<strong>Marginalization property: marginal of a joint Gaussian is Gaussian</strong>
<br>
Let $X = (x_1, ..., x_n)$ and $Y = (y_1, ..., y_n)$

be jointly distributed Gaussian random variables:

$$
p(X, Y) = \mathcal{N} \Bigg(
\left[ \begin{array}{c}{\mu_X} \\ {\mu_Y}\end{array}\right] \, , \,

\left[ \begin{array}{cc}{\Sigma_{XX}} & {\Sigma_{XY}} \\ {\Sigma_{YX}} & {\Sigma_{YY}} \end{array}\right]
\Bigg)
$$

then the marginal distribution is Gaussian

<div class="math">
\begin{align*}

p(X) &= \int p(X, Y) \, dY \\[5pt]
&= \mathcal{N}(\mu_X , \Sigma_{XX})
\end{align*}
</div>

<strong>Conditioning property: conditional of a joint Gaussian is Gaussian</strong>
<br>
It is also the case that:

$$X | Y \sim \mathcal{N}\left(\mu_{X}+\Sigma_{X Y} \Sigma_{YY}^{-1}\left(Y - \mu_{Y}\right), \Sigma_{X X}-\Sigma_{X Y} \Sigma_{YY}^{-1} \Sigma_{Y X}\right) $$

and so conditioning also gives us a Gaussian.
<br>
<br>
</blockquote>

Both of these results will play a key role in Gaussian processes and the use of them will be highlighted when they are encountered. For references on how to derive the above see CS229 notes [here](http://cs229.stanford.edu/section/more_on_gaussians.pdf).

##### Functions as vectors and kernel functions

Explain link to covariance function and producing a PSD matrix which we can sample from. Explain how Cholesky helps us sample from a multivariate Gaussian where the entries are correlated (which we want them to be!).

Use this link: https://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w7b_gaussian_processes.html

##### Combining the mean of Gaussians

The mean of the joint distribution relies on the fact that the resulting mean, $\mu$ of the product of two Gaussian densities with the same means (i.e. $\mu_1 = \mu_2$) is $\mu = \mu_1 = \mu_2$ from:

<div class="math">
\begin{align*}
\mu &= \frac{\sigma_{1}^{-2} \mu_{1}+\sigma_{2}^{-2} \mu_{2}}{\sigma_{1}^{-2}+\sigma_{2}^{-2}} \\[5pt]
\end{align*}
</div>

this is regardless of the values of $\sigma_{1}$ and $\sigma_{2}$ (factor the above).

###### Explain noise term



<hr class="with-margin">
<h4 class="header" id="appendix">Q and A</h4>
<a name="col_rvs"></a>
##### What is a collection of random variables?

Talk about stochastic processes.

<a name="mean_modelling"></a>
##### What does it mean that GPs can model the mean arbitrarily well?

See: https://stats.stackexchange.com/questions/222238/why-is-the-mean-function-in-gaussian-process-uninteresting

Put image of prior dragging function to 0.

##### What is the kernel?

The kernel is actually the crucial thing that determines what sort of functions we end up with. It basically controls the smoothness and type of functions we can get from the prior. We are not going to discuss kernels in this post as it's already too long.

How do I choose the kernel? Read [this](http://www.cs.toronto.edu/~duvenaud/cookbook/index.html).

<a name="know_uncertainty"></a>
##### Why is knowing the uncertainty important?

Can tell us where to sample next if experiments are costly. In high dimensions it takes many function  evaluations to be certain everywhere.

##### How can we simulate from a multivariate Gaussian

See Rasmussen pg 201 - cholesky



##### Why do we assume correlation between observations?

https://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w7b_gaussian_processes.html

We certainly can’t use a diagonal covariance matrix: if our beliefs about the function values are independent, then observing the function in one location will tell us nothing about its values in other locations. We usually want to model continuous functions: so function values for nearby inputs should have large covariances.

##### Why are GPs non-parametric?

Kernel parameters are hyper-parameters. Technically we integrate out all other (infinite) parameters. See Richard Turner talk.

<hr class="with-margin">
<h4 class="header" id="further">References</h4>

Details on joint Gaussians [here](http://cs229.stanford.edu/section/more_on_gaussians.pdf).

http://etheses.whiterose.ac.uk/9968/1/Damianou_Thesis.pdf

The main source I used was [Machine Learning: A Probabilistic Perspective](https://www.amazon.co.uk/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020) by Kevin Murphy which I found more accessible than the authority on the subject: [Rasmussen and Williams](http://www.gaussianprocess.org/gpml/).

http://mlg.eng.cam.ac.uk/teaching/4f13/1819/gaussian%20process.pdf

For more random internet links see [here](http://people.ee.duke.edu/~lcarin/David1.27.06.pdf), [here](https://www.linkedin.com/pulse/machine-learning-intuition-gaussian-processing-chen-yang) and [here](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture2.pdf).

Preliminary reading [here](http://katbailey.github.io/post/gaussian-processes-for-dummies/), [here](http://platypusinnovation.blogspot.com/2016/05/a-simple-intro-to-gaussian-processes.html) and [here](http://keyonvafa.com/gp-tutorial/). There are also good lectures [here](https://www.youtube.com/watch?v=4vGiHC35j9s&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6&t=0s&index=9) from none other than Nando de Freitas that cover Gaussian Processes a lot more thoroughly.

Bay Opt

http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
<hr class="with-margin">
