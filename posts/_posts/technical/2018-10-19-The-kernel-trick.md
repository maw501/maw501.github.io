---
layout: post
title: The kernel trick
date: 2018-10-19
use_math: true
image: "kernel.png"
comments: true
tags: ['kernels', 'SVMs']
---
In this article our goal will be to explain the kernel trick via a simple motivating example which we will follow through in detail to get to the heart of the trick.
<!--more-->
<hr class="with-margin">

<div class="list-of-contents">
  <h4>Contents</h4>
  <ul></ul>
</div>
<hr class="with-margin">
<h4 class="header" id="prereq">Prerequisites</h4>
From my experience kernels often get a bad reputation as they are usually poorly explained, usually in the context of SVMs. This article will focus on the kernel trick, it is therefore helpful to have knowledge of the following topics:

###### Kernel functions
A kernel function $ \textbf{k}$ takes in two data points $x, y \, \in \mathbb{R}^d$ and returns a single number expressing how similar the data points are - the function is symmetric. Formally this can be written as:

$$ \textbf{k}(x, y) : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R} $$

###### SVMs
A support vector machine is a discriminative classifier defined by a separating hyperplane - some notes [here](../../../../project_edx_ml/2018/10/25/columbiaX-ML-week6). It would also be helpful to have knowledge of linear separability and how certain transformations of our data may allow a hyperplane in the new feature space where it wasn't possible in the original space as shown below.

<p align="center">
    <img src="/assets/img/kernel_trick.png" alt="Image" width="500" height="300" />
</p>

<em class="figure">Fig. 0: mapping data to a new feature space can often allow easier separability between classes, [image credit](https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f).</em>

###### Maths
Some basic linear algebra would be useful, pretty much only the [dot product](https://en.wikipedia.org/wiki/Dot_product).

<hr class="with-margin">
<h4 class="header" id="intro">Motivating example</h4>

Let us start by imagining we have a simple dataset with two columns which represent the height and weight of a set of people. For notational simplicity let's call the column representing height $x_1$ and the column representing weight $x_2.$

Further, imagine we are trying to model some target variable (e.g. binary classification as an adult or not) as a function of this data and that we can't solve this problem well with linear combinations of our columns. In other words our data might not be well separated by a linear decision boundary and so we either need a more powerful model or better features.

Now, our friend Bob pops over and suggests we might be able to solve the problem if we take some non-linear transformations of our data. In particular he suggests that we enhance our original dataset to create the following new columns:
  * A column with all 1s in it
  * A column equal to original $x_1$ height feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_1$
  * A column equal to original $x_2$ weight feature multiplied by $\sqrt{2}$, i.e. $\sqrt{2}x_2$
  * A column equal to $\sqrt{2}x_1x_2$
  * A column equal to $x_1^2$
  * A column equal to $x_2^2$

<blockquote class="tip">
<strong>Recall:</strong> SVMs rely on the notion of similarity between each data point in order to locate the most similar points that are in different classes - the support vectors. The goal is then to fit a hyperplane between these support vectors.
</blockquote>

Bob also tells us we can assess how similar two of our data points are (i.e. two rows of this new matrix) by element-wise multiplying the row vectors of our data together and summing the resultant numbers - this is computing the dot product between two points.

We decide to try this by hand.

<p align="center">
    <img src="/assets/img/svm_calc.jpg" alt="Image" width="600" height="400" />
</p>

<em class="figure">Fig. 1: Expanding our original dataset with new features (note it isn't clear yet why we use these new features)</em>

The above is:

$\textbf{a}^T = (a_1, a_2) = (1, 2)$ giving entries:

$$(1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2) = (1, \sqrt{2}, 2\sqrt{2}, 2\sqrt{2}, 1, 4)$$

$\textbf{b}^T = (b_1, b_2) = (3, 4)$ giving entries:

$$(1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2) = (1, 3\sqrt{2}, 4\sqrt{2}, 12\sqrt{2}, 9, 16)$$

If we compute the dot product of these two new rows we get 144 as shown in Fig. 1.

Another friend Alice pops over and sees what we are doing. She laughs that we bothered creating all those new columns and tells us there is a simpler way to get the same answer. She says we should try just computing $(1 + \textbf{a}^T\textbf{b})^2 \,$ directly instead with our original data and not to bother with Bob's idea.

We are a bit sceptical about what Alice means (how can we possibly get the same answer?!) but we decide to try this anyway. Sure enough...

$$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} 1 & 2 \end{array} \right) \left( \begin{array}{c} 3 \\ 4 \end{array} \right) \bigg)^2 = (1 + 11)^2 = 144$$

##### What just happened?!

The kernel trick just happened, that's what.

<hr class="with-margin">
<h4 class="header" id="expl"> Explanation</h4>

<blockquote class="tip">
<strong>Spoiler:</strong> The kernel trick is when we can take a shortcut such as Alice suggested. This means we can perform a calculation between rows of our original data that is equivalent to having created a bunch of new features and performed a calculation with the newly created data. Such kernels are of interest to us when the equivalent long-winded expansion (as suggested by Bob) is a basis for a more complex transformation.
</blockquote>

Alice sees our confusion and decides to explain to us why this worked.

We can call our new features some transformation of our original data, $\phi(\textbf{a})$, so:

$$\phi(\textbf{a}) = (1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2)$$

and similary for row $\textbf{b}$:

$$\phi(\textbf{b}) = (1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2)$$

The dot product between $\phi(\textbf{a})$ and $\phi(\textbf{b})$ gives:

$$\phi(\textbf{a})^T\phi(\textbf{b}) = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$$

This is mathematically identical to

$$(1 + \textbf{a}^T\textbf{b})^2 = \bigg(1 + \left( \begin{array}{c} a_1 & a_2 \end{array} \right) \left( \begin{array}{c} b_1 \\ b_2 \end{array} \right) \bigg)^2 = (1 + a_1b_1 + a_2b_2)^2$$

and expanding the bracket gives

$$(1 + \textbf{a}^T\textbf{b})^2  = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$$

We have thus shown that

$$\phi(\textbf{a})^T\phi(\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2 $$

for some transformation $\phi$ from our original space (which had vectors in 2d) to a feature space over polynomials of the original variables of degree 2. This is called the [quadratic kernel](https://en.wikipedia.org/wiki/Polynomial_kernel).

Weighing up what we have done there are two choices:

1. Start creating numerous columns as per Bob's suggestion to get $\phi(\textbf{a})$ and  $\phi(\textbf{b})$. This will create a large amount of extra columns if we start with more than 2 initially and want a polynomial kernel of high order.
2. Compute $(1 + \textbf{a}^T\textbf{b})^2 = \textbf{k}(\textbf{a},\textbf{b})$ instead - we call this the kernel between two rows/data points.

It's clear that it's computationally much easier to work with our original data and not start creating new columns.

##### Further explanation

<blockquote class="tip">
<strong>Remember:</strong> The special functions for which it turns out allow us to stay in our original lower dimensional space but are equivalent to operating in a higher dimensional space are called kernel functions.
</blockquote>

The **kernel trick** is that we do not need to use the transformation $\phi$ and can actually just stick with our original data and compute $\textbf{k}(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$ instead. The fact we don't need to perform the transformation to a higher dimensional space by creating new columns saves us a lot of computational overhead. This allows us to find complex non linear boundaries that are able to better separate the classes in our dataset.

For the kernel trick to work we must be able to write the calculation in the new higher dimensional feature space as dot products e.g. $\phi(\textbf{a})^T\phi(\textbf{b}).$ This feature space is implicit, and often infinite dimensional.

##### Why do the $\sqrt{2}$s appear?

Consider:

$$\textbf{k}(\textbf{a},\textbf{b}) = (1 + \textbf{a}^T\textbf{b})^2$$

recalling that $\textbf{a}^T = (a_1, a_2)$ and $\textbf{b}^T = (b_1, b_2)$. We have already shown above that this is

$$\textbf{k}(\textbf{a},\textbf{b}) = 1 + 2a_1b_1 + 2a_2b_2 + 2a_1a_2b_1b_2 + a_1^2b_1^2 + a_2^2b_2^2$$

In order to match the 2s that appear in front of the terms above our $\phi$ transformations needed to have $\sqrt{2}$s in the appropriate places:

$$\phi(\textbf{a}) = (1, \sqrt{2}a_1, \sqrt{2}a_2, \sqrt{2}a_1a_2, a_1^2, a_2^2)$$

$$\phi(\textbf{b}) = (1, \sqrt{2}b_1, \sqrt{2}b_2, \sqrt{2}b_1b_2, b_1^2, b_2^2)$$

This is so the terms match when we compute $\phi(\textbf{a})^T\phi(\textbf{b})$. This is essentially what ensures $\phi$ is the correct transformation such that

$$\textbf{k}(\textbf{a},\textbf{b}) = \phi(\textbf{a})^T\phi(\textbf{b})$$

holds.

<hr class="with-margin">
<h4 class="header" id="mercer"> Creating kernels</h4>

The discussion above doesn't tell us how to obtain functions $\textbf{k}(\textbf{a},\textbf{b})$ which are valid kernels. Luckily there is some simple arithmetic that allows us to create new kernels. Let $\textbf{k}_1$ and $\textbf{k}_2$ be any kernels, then constructing $\textbf{k}$ in the following (non-exhaustive) ways results in a new kernel:

* $\textbf{k}(\textbf{a},\textbf{b}) = \textbf{k}_1(\textbf{a},\textbf{b}) \, \textbf{k}_2(\textbf{a},\textbf{b})$
* $\textbf{k}(\textbf{a},\textbf{b}) = \textbf{k}_1(\textbf{a},\textbf{b}) \, + \, \textbf{k}_2(\textbf{a},\textbf{b})$
* $\textbf{k}(\textbf{a},\textbf{b}) = \exp\\{\textbf{k}_1(\textbf{a},\textbf{b})\\}$

In other words we can multiply, add and take the exponential of any existing kernels and still end up with a valid kernel. In these cases we may end up not knowing what the equivalent $\phi$ is, but this doesn't matter as we don't actually need it.

C.M. Bishop, (2006) Pattern Recognition and Machine Learning has a more comprehensive set of ways to construct new kernels.

<hr class="with-margin">
<h4 class="header" id="further">Further reading/videos</h4>

Good video [here](https://www.youtube.com/watch?v=XUj5JbQihlU&hd=1), reading [here](https://stats.stackexchange.com/questions/80398/how-can-svm-find-an-infinite-feature-space-where-linear-separation-is-always-p) and [here](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is).
<hr class="with-margin">
