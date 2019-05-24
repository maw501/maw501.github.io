---
layout: post
title: The kernel trick
date: 2018-10-19
use_math: true
image: "kernel.png"
comments: true
tags: ['kernels', 'SVMs']
---
The kernel trick allows us to implicitly perform a calculation in a high-dimensional feature space whilst only working in the original feature space. In this article we explain the kernel trick via a toy example which we follow through in detail to get to the heart of the trick.
<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>

<hr class="with-margin">
<h4 class="header" id="intro">Introduction</h4>

The kernel trick is a useful technique that allows the use of linear models to model complex non-linear functions by pushing the original data through a non-linear (and possibly high-dimensional) mapping at little extra computational cost. In this way the kernel trick gives us the benefit of using a more powerful non-linear model for the price of sticking with a linear model.

It is important to note that the class of algorithms that kernels work for are based on computing similarity between observations of the data. In other words, we are working with the rows of the data frame for a typical machine learning problem with $n$ rows and $d$ features. The kernel trick is most famously applied to support vector machines (SVMs) for classification problems.

The key benefit to performing such transformations, $\phi$, of the observations, is that after applying the transformation the data can often be linearly separable in the new feature space.

<p align="center">
    <img src="/assets/img/kernel_trick.png" alt="Image" width="500" height="300" />
</p>

<em class="figure">Mapping data to a new feature space can often allow easier separability between classes
<br> [Image credit](https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f)</em>

<hr class="with-margin">
<h4 class="header" id="motivating">Walking through an example</h4>

##### Problem definition
Imagine we have a dataset, $X$, with only two columns which represent the height and weight of a set of people where the target variable, $y$, is a binary number denoting if the person is an adult or not. $X$ has dimensions $n \times d$ where $d=2$ in this example.

This is a binary classification problem and let us further assume that we can't separate the classes well with linear combinations of the columns. In other words the data might not be well separated by a linear decision boundary and so we either need a more powerful model or better features.

For notational simplicity let's call the column representing height $x_1$ and the column representing weight $x_2.$

##### Feature engineering approach

Now, a friend Bob pops over and suggests we might be able to solve the problem if we take some non-linear transformations of the data. In particular he suggests that we enhance the original dataset to create the following new columns:
  * A column with all 1s in it
  * A column equal to the original column for height, $x_1$, multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_1$
  * A column equal to the original column for weight, $x_2$, multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_2$
  * A column equal to $\sqrt{2}x_1x_2$
  * A column equal to $x_1^2$
  * A column equal to $x_2^2$

Bob also tells us we can assess how similar two of the data points are (i.e. two rows of this new matrix) by element-wise multiplying the row vectors of the data together and summing the resultant numbers - this is computing the [dot product](https://en.wikipedia.org/wiki/Dot_product) between two points.

<blockquote class="tip">
<strong>Context:</strong> the notion of similarity between data-points is commonly used across machine learning, in particular in support vector machines (SVMs). In the classification context this measure of similarity is used by SVMs in order to locate points that look similar but are in different classes - these are called the support vectors. The goal is then to fit a hyperplane between these support vectors to separate the classes.
</blockquote>

We decide to take Bob at his word and try this by hand for two rows of the data, $\textbf{a}^T$ and $\textbf{b}^T$.

$\textbf{a}^T = (a_1, a_2) = (1, 2)$ giving new entries for row 1 of:

$$(1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) = (1, \sqrt{2}, 2\sqrt{2}, 2\sqrt{2}, 1, 4)$$

$\textbf{b}^T = (b_1, b_2) = (3, 4)$ giving new entries for row 2 of:

$$(1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2) = (1, 3\sqrt{2}, 4\sqrt{2}, 12\sqrt{2}, 9, 16)$$

If we compute the dot product of these two new rows we get 144:

<div class="math">
\begin{alignat*}{1}
\left(1, \sqrt{2}, 2\sqrt{2}, 2\sqrt{2}, 1, 4 \right)  \left(
\begin{array}{c}
1\\
3\sqrt{2}\\
4\sqrt{2}\\
12\sqrt{2}\\
9\\
16\\
\end{array}
\right)
&= 144.
\end{alignat*}
</div>

##### A shortcut?

Another friend Alice pops over and sees what we are doing. She laughs that we bothered creating all those new columns and tells us there is a simpler way to get the same answer. She says we should try just computing $(1 + \textbf{a}^T\textbf{b})^2 \,$ directly instead with the original data and not to bother with Bob's idea.

We are a bit sceptical about what Alice means (how can we possibly get the same answer?!) but we decide to try this anyway. Sure enough...

$$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} 1 & 2 \end{array} \right) \left( \begin{array}{c} 3 \\ 4 \end{array} \right) \bigg)^2 = (1 + 11)^2 = 144$$

##### What just happened?!

The kernel trick just happened, that's what.

<p align="center">
    <img src="/assets/img/kernel_git_joke.png" alt="Image" width="300" height="60" />
</p>

<hr class="with-margin">
<h4 class="header" id="expl"> Explanation</h4>

<blockquote class="tip">
<strong>Spoiler:</strong> the kernel trick is when we can take a shortcut such as Alice suggested. This means we can perform a calculation between data-points with the original features that is equivalent to having created a larger set of features and taking the dot product between the data-points with the newly engineered features.
</blockquote>

Alice sees the confusion and decides to explain to us why this worked.

Let's call the new features some transformation of the original data, $\phi(\textbf{a}^T)$ and $\phi(\textbf{b}^T)$, so:

<a name="eq0"></a>
<div class="math">
\begin{alignat*}{1}

\phi(\textbf{a})^T &= (1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) \\[5pt]
\phi(\textbf{b})^T &= (1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2) \tag{0}

\end{alignat*}
</div>


The dot product between $\phi(\textbf{a})^T$ and $\phi(\textbf{b})$ is just element wise multiplication and then summing up to give:

$$
\phi(\textbf{a})^T\phi(\textbf{b}) = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2. \tag{1}
$$

This is mathematically identical to

<a name="eq2"></a>
<div class="math">
\begin{alignat*}{1}

(1 + \textbf{a}^T\textbf{b})^2 &= \bigg(1 + \left( \begin{array}{c} a_1 & a_2 \end{array} \right) \left( \begin{array}{c} b_1 \\ b_2 \end{array} \right) \bigg)^2 = (1 + a_1b_1 + a_2b_2)^2 \\[5pt]
&= 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2 \tag{2}

\end{alignat*}
</div>

where equation (1) is the same as (2). We have thus shown that

$$\phi(\textbf{a})^T\phi(\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2 $$

for the transformation $\phi$, as defined in [(0)](#eq0), from the original feature space to a feature space of polynomials with degree 2. This is called the [quadratic kernel](https://en.wikipedia.org/wiki/Polynomial_kernel).

##### Which is easier to compute?

Weighing up what we have done there are two choices:

1. Start creating numerous columns as per Bob's suggestion to get $\phi(\textbf{a})^T$ and  $\phi(\textbf{b})^T$. This will create a large amount of extra columns if $d$ is large and we want a polynomial kernel of high order.
2. Compute $(1 + \textbf{a}^T\textbf{b})^2 = \kappa(\textbf{a},\textbf{b})$ instead.

We call $\kappa(\textbf{a},\textbf{b})$ the kernel between two observations/data-points and it is computationally much easier to work with the original data via option 2 than to start creating many new columns.

##### Further explanation

The kernel trick is that we do not need to use the transformation $\phi$ and can actually just stick with the original data and compute $\kappa(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ instead. The fact we don't need to perform the transformation to a higher dimensional space by creating new columns saves us a lot of computational overhead. This allows us to find complex non linear boundaries that are able to better separate the classes in the dataset.

The special functions for which it turns out allow us to stay in the original lower dimensional space but are equivalent to operating in a higher dimensional space are called kernel functions.

<blockquote class="tip">
<strong>Key takeaway:</strong> the kernel trick is applicable anywhere we are taking dot products between data-points. In order to apply the kernel trick and map the data to a non-linear high-dimensional space we simply replace the original dot product in the algorithm, $\textbf{a}^T \textbf{b}$ with a call to a kernel function, $\kappa( \textbf{a}, \textbf{b})$.
</blockquote>

<hr class="with-margin">
<h4 class="header" id="mercer"> Creating kernels</h4>

The discussion above doesn't tell us how to obtain functions $\kappa(\textbf{a},\textbf{b})$ which are valid kernels. Luckily there is some simple arithmetic that allows us to create new kernels. Let $\kappa_1$ and $\kappa_2$ be any kernels, then constructing $\kappa$ in the following (non-exhaustive) ways results in a new kernel:

* $\kappa(\textbf{a},\textbf{b}) = \kappa_1(\textbf{a},\textbf{b}) \, \kappa_2(\textbf{a},\textbf{b})$
* $\kappa(\textbf{a},\textbf{b}) = \kappa_1(\textbf{a},\textbf{b}) \, + \, \kappa_2(\textbf{a},\textbf{b})$
* $\kappa(\textbf{a},\textbf{b}) = \exp\\{\kappa_1(\textbf{a},\textbf{b})\\}$

In other words we can multiply, add and take the exponential of any existing kernels and still end up with a valid kernel. In these cases we may end up not knowing what the equivalent $\phi$ is, but this doesn't matter as we don't actually need it. Below we introduce two of the most common kernels.

##### General polynomial kernel

So far we have considered a kernel of the form $\kappa(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ which we called a quadratic kernel. A more general form for a polynomial of any order is given by:

$$
\kappa(\textbf{a},\textbf{b}) = (c + \textbf{a}^T\textbf{b})^d \tag{3}
$$

where $d$ is the degree of the polynomial and $c \geq 0$ is a free parameter often set to 1. Varying $c$ amounts to trading off the influence of higher-order versus lower-order terms in the polynomial.

##### Radial basis function kernel

Perhaps the most popular kernel in machine learning is the radial basis function (RBF) kernel, defined as:

$$
\kappa\left(\textbf{a}, \textbf{b}\right)=\exp \left(-\frac{\left\|\textbf{a}-\textbf{b}\right\|^{2}}{2 \sigma^{2}}\right) \tag{4}
$$

with $\sigma^{2}$ a hyperparameter.

##### More on kernels

We have detailed a few simple ways to create kernels but [PRML](#prml) has a more comprehensive guide. A list of common kernels can be found [here](http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#kernel_functions).

<hr class="with-margin">
<h4 class="header" id="prereq"> Helpful prerequisites</h4>
In order to understand the kernel trick it is helpful to have knowledge of the following topics.

##### Kernel functions
A kernel function $ \kappa$ takes in two data points $\textbf{a}, \textbf{b} \, \in \mathbb{R}^d$ and returns a single number expressing how similar the data points are. Formally this can be written as:

$$ \kappa(\textbf{a}, \textbf{b}) : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R} $$

 Kernel functions are symmetric, $\kappa(\textbf{a}, \textbf{b}) = \kappa(\textbf{b}, \textbf{a})$ and so computing the similarity between all the data-points results in a $n \times n$ matrix which is [positive definite](https://en.wikipedia.org/wiki/Definiteness_of_a_matrix).

##### Linear separability
It is helpful to have an understanding of how certain transformations of the data may allow a hyperplane to linear separate the data in a new feature space (right) where it wasn't possible in the original space (left) as shown below.

<p align="center">
    <img src="/assets/img/linear_sep.jpeg" alt="Image" width="700" height="300" />
</p>

<em class="figure">Knowledge of the distance from the origin makes this data linearly separable</em>

<hr class="with-margin">
<h4 class="header" id="references">References</h4>

<a name="prml"></a>
* Bishop, C. (2006). [Pattern Recognition and Machine Learning](https://www.springer.com/gb/book/9780387310732), Chapter 6

<hr class="with-margin">
<h4 class="header" id="appendix">Appendix: Q and A</h4>

<a name="root2s"></a>
##### Why do the $\sqrt{2}$s appear in some $\phi$ terms?

Recall the $\phi$ expansions as:

<div class="math">
\begin{alignat*}{1}

\phi(\textbf{a}^T) &= (1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) \\[5pt]
\phi(\textbf{b}^T) &= (1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2)

\end{alignat*}
</div>

and that for $\textbf{a}^T = (a_1, a_2)$ and $\textbf{b}^T = (b_1, b_2)$, we showed above in [(2)](#eq2) that:

<div class="math">
\begin{alignat*}{1}

\kappa(\textbf{a},\textbf{b}) &= (1 + \textbf{a}^T\textbf{b})^2\\[5pt]
&= 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2  + a_1^2b_1^2 + a_2^2b_2^2. \tag{5}
\end{alignat*}
</div>

In order to match the factors of 2 that appear in front of the terms in (5) the transformation $\phi$ requires $\sqrt{2}$s in the appropriate places. This ensures when we compute $\phi(\textbf{a})^T\phi(\textbf{b})$ the factors match and $\phi$ is the correct transformation such that:

$$\kappa(\textbf{a},\textbf{b}) = \phi(\textbf{a})^T\phi(\textbf{b}).$$

<hr class="with-margin">
