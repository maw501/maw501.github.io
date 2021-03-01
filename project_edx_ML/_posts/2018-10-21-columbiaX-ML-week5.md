---
layout: post
title: ColumbiaX - ML - week 5
date: 2018-10-21
use_math: true3456
image: "logreg.png"
comments: true
tags: [classification, logistic regression, kernels]
#github_repo: https://github.com/maw501/edx_machine_learning/tree/master/week5
---
We discuss logistic regression, a discriminative linear classification model whose Bayesian extension requires the Laplace approximation technique for approximating the posterior distribution. We then move onto discussing feature expansions and kernel methods.
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

<a name="intro"></a>
<hr class="with-margin">
<h4 class="header" id="intro">Overview</h4>

We start by introducing [logistic regression](#log_reg), motivated by the previous work we have done on linear classifiers and hyperplanes. This is then extended into a Bayesian setting to give [Bayesian logistic regression](#bayes_log_reg). Calculating the posterior for Bayesian logistic regression with a $l_2$ prior on the weight vector is analytically intractable and so an approximation is used. This approximation uses a Gaussian distribution centred at the MAP estimate for the posterior and is based on [Laplace's method](https://en.wikipedia.org/wiki/Laplace%27s_method). Whilst this calculation is not of central interest for machine learning we spend time in the [appendix](#laplace_approx) extending this calculation by breaking it down into easier to follow steps.

We then move to talking briefly about [feature expansions and kernels](#kernels) and it is here we meet the major deviation from the content as presented in the lectures. Gaussian processes are omitted entirely and instead the reader is referred to a more comprehensive (and hopefully accessible) treatment in a [separate post](/../posts/2019/05/08/Gaussian-Processes) I have written. This is also the case for the [kernel trick](/../posts/2018/10/19/The-kernel-trick).

The net result of these omissions and extensions leaves the appendix comparatively large compared to the content however the flow is hopefully more natural because of it.

<a name="log_reg"></a>
<hr class="with-margin">
<h4 class="header" id="log_reg">Logistic regression</h4>

##### Introduction

Logistic regression, despite its name, is an algorithm that can be used for binary and multi-class classification. There are several popular ways of motivating the introduction of logistic regression and we do so here by continuing from [last week's](/../project_edx_ml/2018/10/16/columbiaX-ML-week4#perceptron) discussion of the perceptron which attempted to fit a hyperplane to data and then classify an example using:

<div class="math">
\begin{align*}
f(\mathbf{x})=\operatorname{sign}\left(w_{0}+\mathbf{x}^{T} \mathbf{w}\right).
\end{align*}
</div>

Where we recall that $\mathbf{x}^{T} \mathbf{w}+w_{0} = 0$ defines the equation of the hyperplane.

Logistic regression combines the idea of a hyperplane with a probabilistic estimate of the classification for each example. It does this in a discriminative manner, directly attempting to model $p(y \| \mathbf{x})$ for every example. This is in contrast with the LDA and QDA models we saw which made explicit assumptions about the data in order to try to model $p(y)$ and $p(\mathbf{x} \| y)$.

In the next section we show how we get to logistic regression from thinking about the log odds interpretation we used for LDA and QDA.

<blockquote class="tip">
<strong>Recap from week 4:</strong> in <a class="reference external" href="/../project_edx_ml/2018/10/16/columbiaX-ML-week4#lda_qda">week 4</a>
we saw how for binary classification with LDA we declared an example to be in a particular class if the log odds were greater than 0:

<div class="math">
\begin{alignat*}{1}
\frac{p(\mathbf{x} | y=1) p(y=1)}{p(\mathbf{x} | y=-1) p(y=-1)} &> 1 \iff \\[5pt]
\underbrace{\ln \frac{p(\mathbf{x} | y=1) p(y=1)}{p(\mathbf{x} | y=-1) p(y=-1)}}_\text{log odds} &> 0 \tag{R1}
\end{alignat*}
</div>

where we are making a notational switch to have $y \in \{-1, 1 \}$ now rather than $y \in \{0, 1 \}$.

For logistic regression the goal is now to directly model $p(y | \mathbf{x})$ rather than use a generative model to model $p(y)$ and $p(\mathbf{x} | y)$ which we did for LDA and QDA.
<br>
<br>
We also
<a class="reference external" href="/../project_edx_ml/2018/10/16/columbiaX-ML-week4#lda_equation">showed</a>
 that evaluating $(\text{R}1)$ led to, for LDA, a decision based on:

<div class="math">
\begin{align*}
f(\mathbf{x})=\operatorname{sign}\left(w_{0}+\mathbf{x}^{T} \mathbf{w}\right).
\end{align*}
</div>

where we had analytic expressions for $w_0$ and $\mathbf{w}$.
</blockquote>

##### Logistic link function

For LDA and QDA we modelled $p(y)$ and $p(\mathbf{x} \| y)$ and then computed the log odds to make a decision where we had analytic expressions for the parameters $w_0$ and $\mathbf{w}$.

We now relax these assumptions and try to model $p(y \| \mathbf{x})$ directly by learning values for $w_0$ and $\mathbf{w}$.

We will still work with the set-up of log odds and a hyperplane decision:

<div class="math">
\begin{alignat*}{1}
\ln \frac{p(y=+1 | \mathbf{x})}{p(y=-1 | \mathbf{x})} &= \mathbf{x}^{T} \mathbf{w}+w_{0} \\[5pt] \tag{A0}
\end{alignat*}
</div>

where $(\text{A}0)$ being positive means we would classify an example as in class $y = +1$ and negative as class $y = -1$.

For binary classification we can now show how the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) arises from thinking about the above set-up in equation $(\text{A}0)$. Recall for binary classification we have:

<div class="math">
\begin{alignat*}{1}
p(y=+1 | \mathbf{x}) &= 1 - (y=-1 | \mathbf{x}) \\[5pt]
\frac{p(y=+1 | \mathbf{x})}{1-p(y=+1 | \mathbf{x})} &= \exp \left\{\mathbf{x}^{T} \mathbf{w}+w_{0} \right\} &\text{exponentiating eqtn. (0)} \\[5pt]
\Rightarrow p(y=+1 | \mathbf{x}) &= \frac{\exp \left\{\mathbf{x}^{T} \mathbf{w}+w_{0}\right\}}{1+\exp \left\{\mathbf{x}^{T} \mathbf{w}+w_{0}\right\}} \hspace{1cm} &\text{rearranging} \\[5pt]
&=\sigma\left(\mathbf{x}^{T} \mathbf{w}+w_{0}\right) \tag{1}
\end{alignat*}
</div>

where we call $\sigma$ the sigmoid function and $\mathbf{x}^{T} \mathbf{w}+w_{0}$ is called the [link function](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function) for the log odds. This is the usual form in which logistic regression is presented without the long build-up we've been through so far.

<blockquote class="comment">
<strong>Comment</strong>
<hr class="small-margin">
Whilst it may seem simple to postulate logistic regression in the above form (i.e. a linear model with some activation pushing the response into the range $[0,1]$ to get a probabilistic interpretation) it's hopefully pleasing to see the background tying in hyperplanes, log odds, LDA and QDA. We will also tie in the perceptron a little closer when we show the algorithm for logistic regression.
</blockquote>

##### Logistic regression likelihood

Similar to the approach we took for linear regression without regularization, solving logistic regression is by maximum likelihood.

Using a [change of notation](#logreg_notation) we can write this joint data likelihood as:

<div class="math">
\begin{alignat*}{1}

p\left(y_{1}, \ldots, y_{n} | \mathbf{x}_1, \ldots, \mathbf{x}_n, \mathbf{w}\right)

&= \prod_{i=1}^{n} \sigma_{i}\left(y_{i} \cdot \mathbf{w}\right)
\end{alignat*}
</div>

where the goal is to find the optimal weight vector $\mathbf{w}\_{ML}$ that maximizes this term. In other words, to get the maximum likelihood solution we need to solve:

<div class="math">
\begin{alignat*}{1}
\mathbf{w}_{ML} &= \arg \max_{\mathbf{w}} \underbrace{\sum_{i=1}^{n} \ln \sigma_{i}\left(y_{i} \cdot \mathbf{w}\right)}_\text{$= \, \mathcal{L}$} \\[5pt]
&=\arg \max_{\mathbf{w}} \mathcal{L}.
\end{alignat*}
</div>

However, unlike linear regression, this cannot be solve analytically, and so we need to use an iterative algorithm like gradient ascent. Finding the derivative of the sigmoid function and its log is straightforward and is given in the [appendix](#sig_deriv), here we just state the results:

<div class="math">
\begin{align*}
\nabla_{z} \sigma(z) = \sigma(z)(1-\sigma(z)), \,\,\,\, \nabla_z \, \ln \sigma(z) = \frac{1}{1 + e^z}
\end{align*}
</div>

which will be used to calculate $\nabla_{\mathbf{w}} \mathcal{L}$.

##### Logistic regression algorithm

We now present a simple form of the logistic regression algorithm with gradient ascent which we then link to the perceptron algorithm.

<blockquote class="algo">
<hr class="small-margin">
<strong>Algorithm: logistic regression</strong>
<hr class="small-margin">
<strong>Input:</strong> training data $(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)$ and step size $\eta$.
<br>
<br>
1. <strong>Initialise:</strong> $\mathbf{w}^{(0)}$ to the zero vector
<br>
2. <strong>For step </strong>1, 2, ..., <strong>do:</strong>
<br>
&emsp; &emsp; Update:

<div class="math">
\begin{align*}
\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}+\eta \underbrace{\sum_{i=1}^{n}\overbrace{\left(1-\sigma_{i}\left(y_{i} \cdot \mathbf{w}\right)\right)}^\text{prob of misclassification} y_{i} \mathbf{x}_{i}}_\text{$= \, \nabla_{\mathbf{w}} \mathcal{L}$} \tag{LRGD}
\end{align*}
</div>
</blockquote>

##### Link to perceptron
Recall from [week 4](/../project_edx_ml/2018/10/16/columbiaX-ML-week4#perceptron_algo) the perceptron update step is:  $\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} +\eta y_i \mathbf{x}_i$ where we only update the observations that are misclassified.

Logistic regression has basically the same update step except the gradient update step weights each example by the probability of assigning an observation to the wrong label, $1-\sigma_{i}\left(y_{i}+\mathbf{w}\right)$. We then sum these probabilities of being wrong over all of our data points and use this to weight the update to $\mathbf{w}^{(t+1)}$. Note we don't have to perform the update over all the data and using batches would give [stochastic gradient ascent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

Thus the logistic regression update step is the same as for the perceptron except for the perceptron an observation is either classified correctly or incorrectly whereas logistic regression has a probability weighting.

It is this probability weighting that actually causes an issue with plain logistic regression as we discuss next.

<a name="bayes_log_reg"></a>
<hr class="with-margin">
<h4 class="header" id="bayes_log_reg">Bayesian logistic regression</h4>

<blockquote class="tip">
<strong>TLDR:</strong> Bayesian logistic regression regularizes the weights by placing a Gaussian prior over them. The posterior is analytically intractable and so a Gaussian approximation to the posterior centred at $\mathbf{w}_{MAP}$ is calculated instead.
</blockquote>

##### Introduction

It turns out that for logistic regression if the data is linearly separable then the weights in the logistic regression optimization can end up becoming very large. In the case of linear regression we regularized the size of the weights by modifying the optimization function (or adding a prior over the weights in a Bayesian sense) and we can do the same for logistic regression. For linear regression this was covered in [weeks 1 and 2](/../project_edx_ml/2018/10/10/columbiaX-ML-week1and2#rr_map).

<blockquote class="tip">
<strong>Explaining why the weights become large when the data is linearly separable:</strong> an intuitive explanation of why this happens is to note that the joint likelihood of the data is maximized when we are perfectly confident about each classification, so $\sigma_{i}\left(y_{i} \cdot \mathbf{w}\right) = 1$ for each $i$.
<br>
<br>
Recall that the hyperplane separating the classes is given by $\mathbf{x}^{T} \mathbf{w} = 0$ and that multiplying this by any constant, $c$, does not change the location of the hyperplane. Thus we can increase the size of every element of $\mathbf{w}$ arbitrarily without moving the hyperplane.
<br>
<br>
How does this effect the likelihood?
<br>
<br>
Well, from the definition of the sigmoid function, if everything is correctly classified then increasing $\mathbf{w}$ will drive the probability predictions further towards their correct answers. In other words we can increase the likelihood without moving the hyperplane by just scaling $\mathbf{w}$.
<br>
<br>
The end result of this, in binary classification, would be to output either probability 1 or 0 for each class prediction.
<br>
<br>
Further reading on this issue is available from <a class="reference external" href="{{page.url}}#ryan_adams">here.</a>
</blockquote>

##### Bayesian logistic regression

As we have mentioned above the natural answer to this issue is to modify the objective function by adding a penalty based on the magnitude of the weight vector. In the case of using a $l_2$ penalty this is the same as putting a Gaussian prior on $\mathbf{w}$.

The objective function for logistic regression now becomes:

<div class="math">
\begin{align*}
\mathbf{w}_{MAP}=\arg \max_{\mathbf{w}} \sum_{i=1}^{n} \ln \sigma_{i}\left(y_{i} \cdot \mathbf{w}\right)-\lambda \mathbf{w}^{T} \mathbf{w} \tag{2}
\end{align*}
</div>

where the maximization problem is now seeking the [MAP estimate](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) for the weights.

Unfortunately when doing this there is no analytic expression for the posterior $p(\mathbf{w} \mid X, \mathbf{y})$ and instead we approximate the posterior using a technique called [Laplace approximation](#laplace_approx). Nearly all the heavy lifting for Bayesian logistic regression is when presenting the Laplace approximation and we have moved the details of this to the appendix.

One point to note about the Laplace approximation for Bayesian logistic regression is that we still need to find $\mathbf{w}\_{MAP}$ via. optimization. Once we have $\mathbf{w}z\_{MAP}$, the posterior is determined analytically.

If you don't want the details the summary of Laplace approximation is [here](#laplace_summary).

<a name="kernels"></a>
<hr class="with-margin">
<h4 class="header" id="kernels">Kernels</h4>

##### Introduction
So far we have talked about algorithms that learn a parameter vector $\mathbf{w}$ based on some training data and at prediction time use this $\mathbf{w}$ to make an estimate for a new data-point. In such methods we no longer require the original training data at prediction time - it essentially can be thrown away.

However there are certain algorithms in which the training data-points, or a subset of them, are required at prediction time. These algorithms usually require computing some sense of similarity between observations of the data. In this sense these algorithms are sometimes called 'memory-based' methods and can be fast to train but slow when making predictions. [$k$-NNs](/../project_edx_ml/2018/10/16/columbiaX-ML-week4#knn) which was introduced last week is one such algorithm.

Computing the similarity between data-points based on a fixed non-linear feature space mapping is based on the use of kernel functions and this is the main focus of this section after briefly mentioning feature expansions.

##### Feature expansions

Feature expansions are a simple idea which every data scientist should be familiar with.

Feature or basis expansions involve taking a transformation, $\phi(\mathbf{x})$, of the features of the dataset usually in order to capture non-linear behaviour or interactions between features. The general idea is for a point $\mathbf{x} \in \mathbb{R}^d$ we map it to a higher dimensional space $\phi(\mathbf{x}) \in \mathbb{R}^D$ where $D > d$. Such transformations can be as simple as the squaring of each of the original columns and interaction terms. We can then fit a linear model to the enhanced set of features.

The key motivation for performing such transformations is that after applying the transformation the data can often be linearly separable in the newly enhanced feature space.

However it's usually not obvious which type of feature expansion to perform. The manual approach to this is often called feature engineering and usually requires domain knowledge or in-depth exploratory data analysis.

Once the new features have been computed we could then fit a model with a $l_1$ penalty in order to find a sparse subset of the enhanced feature space.

##### Kernels

We start by defining a kernel function before tying it into the discussion above on feature expansions.

<blockquote class="math">
<strong>Definition: kernel</strong>
<hr class="small-margin">
A kernel $\kappa(\cdot, \cdot) : \mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$ is a symmetric function defined such that for any set of $n$ data-points $(\mathbf{x}_1, ..., \mathbf{x}_n ) \in \mathbb{R}^d $ the $n \times n$ matrix $K$ with $K_{i j} = \kappa\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)$ is
<a class="reference external" href="https://en.wikipedia.org/wiki/Definiteness_of_a_matrix">positive semi-definite</a>.
<br>
<br>
Note: $\kappa(\cdot, \cdot)$ is used to denote the function taking 2 inputs.
<br>
<br>
Intuitively, this means $K$ satisfies the properties of a covariance matrix, though note that in general positive semi-definite matrices can have negative entries.
</blockquote>

Based on [Mercer's theorem](#mercer) if $\kappa$ satisfies the above definition then we have that:

<div class="math">
\begin{align*}
\kappa\left(\mathbf{x}_{i}, \mathbf{x}_{j}\right)=\phi\left(\mathbf{x}_{i}\right)^{T} \phi\left(\mathbf{x}_{j}\right). \tag{3}
\end{align*}
</div>

In words the above is saying that there are certain functions, $\kappa$ that are the equivalent to computing the dot product between data-points in some feature space $\phi : \mathbb{R}^d \rightarrow \mathbb{R}^D$. The reason this is interesting and of use is that computing $K$ using $\kappa$ is often much more efficient than using $\phi$ due to what is commonly referred to as the kernel trick. Given the importance and amount of times this crops up I have written a [separate post](/../posts/2018/10/19/The-kernel-trick) on it.

Kernels are useful as many machine learning algorithms rely on the computation of the dot product between data-points. Thus if an algorithm is formulated such that data-points $\mathbf{x}$ occur only in the form of dot products then we can replace this dot product with some choice of kernel, which is the same as having used a transformation $\phi(\mathbf{x})$.

This essentially allows us to lift the calculation into a higher-dimensional feature space, often at little extra computational cost (due to the kernel trick).

<blockquote class="tip">
<strong>TLDR:</strong> generally all the above amounts to in practice is replacing $\mathbf{x}_i^T \mathbf{x}_i$ everywhere in an algorithm’s definition with $\kappa\left(\mathbf{x}_i, \mathbf{x}_j\right)=\phi\left(\mathbf{x}_i\right)^T \phi\left(\mathbf{x}_j\right)$. This allows us to enhance many well-known algorithms.
</blockquote>

##### Dual representation
In order to take advantage of kernels it turns out we can actually reformulate many linear parametric models into an equivalent so called dual representation whereby the predictions are based on linear combinations of a kernel function evaluated at the training data points.

For example, in [PRML [6.1]](#prml), it is shown that we can rewrite the prediction from ridge regression for a new test point $\mathbf{x}\_{\star}$, $y\_{\star}$, as:

<div class="math">
\begin{align*}
y_{\star} = \underbrace{\mathbf{k}_{\star}^T}_\text{$1 \times n$}\underbrace{(K + \lambda I)^{-1}}_\text{$n \times n$}\underbrace{\textbf{y}}_\text{$n \times 1$} \tag{4}
\end{align*}
</div>

where $\mathbf{k}\_{\star} = \kappa(\mathbf{x}\_{\star}, X)$ is the vector with the kernel function computed for a single test point against all the training data and $K_\{i j} = \kappa\left(\mathbf{x}_i, \mathbf{x}_j\right)$ is an $n \times n$ matrix with the kernel function evaluated for all the training data.

The important thing to note is that the prediction at $\mathbf{x}\_{\star}$ is a linear combination of the target values from the training set.

Why would we do this?

Consider the prediction from ridge regression if we instead computed a feature expansion $\phi$ of the data:

<div class="math">
\begin{align*}
y_{\star} = \mathbf{w}^T \phi(\mathbf{x}_{\star}).
\end{align*}
</div>

Using kernels avoids the explicit introduction of the feature transformation, $\phi$, which means we can implicitly use feature spaces of high, even infinite, dimensionality. Of course, this means we pay a price in terms of inverting a $n \times n$ matrix vs. one of $d \times d$ which is required in the solution for the solution to ridge regression, $\mathbf{w}\_{RR}$. This solution, $\mathbf{w}\_{RR}$, is covered in [weeks 1 and 2](/../project_edx_ml/2018/10/10/columbiaX-ML-week1and2#rr_map) of the course.

This duality allows us to use the kernel trick to expand the representational power of a model often without much greater computational expense.

The field of kernels and the types we can use to model the data are many and varied and below we introduce a single one. For further reading see [PRML [6.2]](#prml).

##### Gaussian kernel (radial basis function)

There are many kernels that can describe different classes of functions, including to encode properties such as periodicity. In this post we will restrict ourselves to the most common kernel which we merely state (as it's extensively written about elsewhere), the [radial basis function (or Gaussian) kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel):

<div class="math">
\begin{align*}
\kappa(\mathbf{x_i}, \mathbf{x_j}) = \sigma^{2} \exp (-\frac{ \| \mathbf{x_i} - \mathbf{x_j} \|^{2}}{2 l^{2}}) \tag{5}
\end{align*}
</div>
with hyperparameters $\sigma$ and $l$.

<hr class="with-margin">
<h4 class="header" id="references">References</h4>
<div class="bullet"> 
<li>
<a name="prml"></a>
Bishop, C. (2006). Chapters: 4.3 - 4.5, 6; <a class="reference external" href="https://www.springer.com/gb/book/9780387310732">Pattern Recognition and Machine Learning</a>.</li>
<li>
<a name="esl"></a>
Hastie, T., R. Tibshirani, and J. Friedman (2001). Chapters: 4.4, 6;  <a class="reference external" href="http://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a>.</li>
<li>
<a name="edx_ml"></a>
edX, ColumbiaX, <a class="reference external" href="https://www.edx.org/course/machine-learning-1">Machine Learning</a>.</li>
</div>
<br>
##### Further reading
<div class="bullet"> 
<a name="ryan_adams"></a>
<li> Adams, R, Princeton University, Spring 2019, Lecture 9: Linear Classification II <a class="reference external" href="https://www.cs.princeton.edu/courses/archive/spring19/cos324/">COS 324: Introduction to Machine Learning</a>.</li>
<li> Neal, R, M <a class="reference external" href="http://www.cs.utoronto.ca/~radford/csc2541.S11/week10.pdf">CSC2541:Bayesian Methods for Machine Learning</a>.</li>
<li> Murray, I, <a class="reference external" href="https://www.inf.ed.ac.uk/teaching/courses/mlpr/2016/notes/w8a_bayes_logistic_regression_laplace.pdf">Bayesian logistic regression and Laplace approximations</a>.</li>
<li> Murphy K, Section 8.4, <a class="reference external" href="https://www.amazon.co.uk/Machine-Learning-Probabilistic-Perspective-Computation/dp/0262018020">Machine Learning: A Probabilistic Perspective</a>.</li>
</div>

<a name="appendix"></a>
<hr class="with-margin">
<h4 class="header" id="appendix">Appendix: Q and A</h4>

<a name="logreg_notation"></a>
##### Notation for logistic regression
It's actually possible to write the equations for logistic regression more compactly using a change of notation and it pays to get familiar with this style as it is commonly used. There are 3 notational changes we make.

###### 1. Absorb intercept
Absorb the $w_0$ term into the weight vector $\mathbf{w}$ which amounts to appending a 1 to each $\mathbf{x}$ as follows:

<div class="math">
\begin{align*}
\mathbf{w} \leftarrow\left[\begin{array}{c}{w_{0}} \\ {\mathbf{w}}\end{array}\right], \, \, \, \mathbf{x} \leftarrow\left[\begin{array}{l}{1} \\ {\mathbf{x}}\end{array}\right]
\end{align*}
</div>

###### 2. Change sigmoid notation
A simple change depending on style, define $\sigma_{i}(\mathbf{w})=\sigma\left(\mathbf{x}_{i}^{T} \mathbf{w}\right)$


###### 3. Use $y_i$ in sigmoid

For the case where $y_i=1$:

<div class="math">
\begin{alignat*}{1}

p\left(y = 1 | \mathbf{x} \right) &= \sigma_{i}(\mathbf{w}) \\[5pt]
&= \left( \frac{\exp\{\mathbf{x}_i^T \mathbf{w}\}} {1+\exp\{\mathbf{x}_i^T \mathbf{w}\}} \right) \\[5pt]

&= \left( \frac{\exp\{y_i \mathbf{x}_i^T \mathbf{w}\}} {1+\exp\{y_i \mathbf{x}_i^T \mathbf{w}\}} \right) \\[5pt]

&= \sigma_i(y_i \mathbf{w})
\end{alignat*}
</div>

For the case where $y_i=-1$:

<div class="math">
\begin{alignat*}{1}

p\left(y = -1 | \mathbf{x} \right) &= 1 - \sigma_{i}(\mathbf{w}) \\[5pt]
&= \left( 1 -  \frac{\exp\{\mathbf{x}_i^T \mathbf{w}\}} {1+\exp\{\mathbf{x}_i^T \mathbf{w}\}} \right) \\[5pt]

&= \left(\frac{1}  {1+ \exp\{\mathbf{x}_i^T \mathbf{w}\}} \right) \\[5pt]
&= \left(\frac{1}  {1+ \exp\{-y_i \mathbf{x}_i^T \mathbf{w}\}} \right) \\[5pt]
&= \sigma_i(y_i \mathbf{w})
\end{alignat*}
</div>

which uses the fact the sigmoid function can be written in two equivalent ways:

<div class="math">
\begin{align*}
\frac{1}{1+e^{-x}}=\frac{e^{x}}{e^{x}+1}
\end{align*}
</div>

and so we can denote $\sigma_i(y_i \mathbf{w})$ as the probability that observation $i$ has label $y_i$ for both $y=1$ or $y=-1.$


<a name="sig_deriv"></a>

##### Derivative of sigmoid function

<blockquote class="math">
<strong>Summary of approach:</strong> we use substitution with the chain rule and also recall that:

<div class="math">
\begin{alignat*}{1}
\sigma(z) &= \frac{1}{1 + e^{-z}} \\[5pt]
\Rightarrow e^{-z} &= \frac{1 - \sigma(z)}{\sigma(z)}
\end{alignat*}
</div>
</blockquote>

Apply the chain rule to proceed:

<div class="math">
\begin{alignat*}{1}
\hspace{5cm} \nabla_z \, \sigma(z) &= \nabla_z \frac{1}{1 + e^{-z}} \hspace{2cm} &\text{by defn.} \\[5pt]
&= \frac{e^{-z}}{(1 + e^{-z})^2} &\text{by chain rule with $u=1+e^{-z}$} \\[5pt]
&= e^{-z}\underbrace{\left(\frac{1}{(1 + e^{-z})}\right)^2}_\text{$\sigma^2(z)$} &\text{split out} \\[5pt]
&= e^{-z} \sigma^2(z) \\[5pt]
&= \sigma(z)(1-\sigma(z)) &\text{using defn. of $e^{-z}$}
\end{alignat*}
</div>

Note that we can similarly show that:

<div class="math">
\begin{alignat*}{1}
\nabla_z \, \ln \sigma(z) &= \frac{1}{1 + e^z}
\end{alignat*}
</div>

<a name="laplace_approx"></a>
##### Laplace approximation
<blockquote class="tip">
<strong>TLDR:</strong> we can't calculate the posterior distribution, $p(\mathbf{w} | X, \mathbf{y})$, for Bayesian logistic regression with an $l_2$ prior analytically so we approximate it with a Gaussian distribution with mean vector centred at the MAP solution for $\mathbf{w}$.
</blockquote>

###### Summary
Laplace approximation is a way of approximating the posterior distribution $p(\mathbf{w} \| X, \mathbf{y})$ analytically by finding a Gaussian approximation to the posterior distribution. The method aims specifically at problems in which the posterior distribution is uni-modal.

The derivation of the below is pretty involved and a summary is provided below. We will endeavour to explain all steps as we walk through the derivation in more detail.

<a name="laplace_summary"></a>
<blockquote class="tip">
<strong>Summary of Laplace approximation for logistic regression</strong>
<br>
Given labeled data $(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n) $ and the model:


<div class="math">
\begin{align*}
\color{#e06c75}{p\left(y_{i} | \mathbf{x}_i, \mathbf{w} \right)=\sigma\left(y_i \mathbf{x}_i^T \mathbf{w}\right)}, \quad
\color{#61afef}{\mathbf{w} \sim \mathcal{N}\left(0, \lambda^{-1} I\right)}, \quad
\color{#98c379}{\sigma\left(y_i \mathbf{x}_i^T \mathbf{w}\right)=\frac{\exp\left\{y_{i} \mathbf{x}_i^T  \mathbf{w}\right\}}{1+\exp\left\{y_{i} \mathbf{x}_i^T  \mathbf{w}\right\}}}
\end{align*}
</div>

where the above is defining <span style="color:#e06c75">the likelihood</span>, <span style="color:#61afef">the prior</span> and <span style="color:#98c379">the definition of sigmoid</span>.
<br>
<br>
<strong>Step 1:</strong> find the MAP solution for the weights by solving the optimization problem given by the objective function of logistic regression:

<div class="math">
\begin{align*}
\mathbf{w}_{MAP}=\arg \max_{\mathbf{w}} \sum_{i=1}^{n} \ln \sigma\left(y_{i} \mathbf{x}_i^T \mathbf{w}\right)-\frac{\lambda}{2} \mathbf{w}^T \mathbf{w}
\end{align*}
</div>

which is typically done via some optimization algorithm.
<br>
<br>
<strong>Step 2:</strong> compute the (inverse) covariance matrix as

<div class="math">
\begin{align*}
-\Sigma^{-1}=-\lambda I-\sum_{i=1}^{n} \sigma\left(y_i \mathbf{x}_i^T \mathbf{w}_{MAP}\right)\left(1-\sigma\left(y_i \mathbf{x}_i^T \mathbf{w}_{MAP}\right)\right) \mathbf{x}_i \mathbf{x}_i^T
\end{align*}
</div>

where details of how we arrive at the above expression are given below.
<br>
<br>
<strong>Step 3:</strong> approximate the posterior using a Gaussian

<div class="math">
\begin{align*}
p(\mathbf{w} | X, \mathbf{y})=\mathcal{N}\left(\mathbf{w}_{MAP}, \Sigma\right)
\end{align*}
</div>

</blockquote>

###### Some prerequisites

Before starting we give some things to watch out for as we proceed.

<blockquote class="math">
<strong>Helpful things to bear in mind</strong>
<br>
<li> The denominator in Bayes' rule can be written as the numerator integrated over the parameter of interest, here $\mathbf{w}$. </li>
<br>
<br>
<li>Bayes' rule can be written just using the joint distribution:

<div class="math">
\begin{align*}
p(\mathbf{w} | \mathbf{y}, X) = \frac{p(\mathbf{y}, \mathbf{w} | X)}{\int p(\mathbf{y}, \mathbf{w} | X) \, d\mathbf{w}}.
\end{align*}
</div>
</li>
<br>
<li>Laplace approximation is based on <a class="reference external" href="https://en.wikipedia.org/wiki/Laplace%27s_method">Laplace's method</a> which approximates integrals of the form:

<div class="math">
\begin{align*}
\int_{a}^{b} e^{M f(x)} d x
\end{align*}
</div>

for a large number $M$ and where $f(x)$ is a twice-differentiable function. Here we forget about $M$ and just rewrite the integral into the required form by exponentiating then taking logs (which cancels out).</li>
<br>

<li>The second order Taylor expansion for a function $f(\mathbf{w})$ with $\mathbf{w} \in \mathbb{R}^{d+1}$ at a point $z \in \mathbb{R}^{d+1}$ is:

<div class="math">
\begin{align*}
f(\mathbf{w}) \approx f(z)+(\mathbf{w}-z)^{T} \nabla f(z)+\frac{1}{2}(\mathbf{w}-z)^{T}\left(\nabla^{2} f(z)\right)(\mathbf{w}-z)
\end{align*}
</div>

where we use the notation that:

<div class="math">
\begin{align*}
\nabla f(z) = \nabla_{\mathbf{w}} f(\mathbf{w})\mid_{z}
\end{align*}
</div>

is the derivative of the function $f$ w.r.t $\mathbf{w}$ evaluated at the point $z$. </li>
</blockquote>

###### Choosing the form of the posterior
The goal is to approximate the posterior with a Gaussian distribution:

<div class="math">
\begin{align*}
p(\mathbf{w} | X, \mathbf{y}) \approx \mathcal{N}(\mu, \Sigma)
\end{align*}
</div>

for some $\mu$ and $\Sigma$ which we need to find. The true posterior distribution is a multivariate distribution in parameter space that will (hopefully) have some dominant mode - we would like to center the Gaussian distribution here if we are able to.

By Bayes' rule and rewriting the joint distribution to get it into a form for Laplace's method we have:

<div class="math">
\begin{alignat*}{1}
p(\mathbf{w} | X, \mathbf{y})
&= \frac{p(\mathbf{y}, \mathbf{w} | X)}{\int p(\mathbf{y}, \mathbf{w} | X) d\mathbf{w} } \\[5pt]
&= \frac{\exp\{\ln p(\mathbf{y}, \mathbf{w} | X)\}}{\int \exp\{\ln p(\mathbf{y}, \mathbf{w} | X)\} d\mathbf{w} }.
\end{alignat*}
</div>

###### Using the second-order Taylor expansion

For Laplace's method we define a function $f$ equal to the log of the joint likelihood:

<div class="math">
\begin{align*}
f(\mathbf{w})=\ln p(\mathbf{y}, \mathbf{w} | X)
\end{align*}
</div>

which we will expand $f(\mathbf{w})$ around a point, $z$. We say $f$ if only a function of $\mathbf{w}$ as this is the variable of interest, the data $\mathbf{y}$ and $X$ are given and considered fixed.

What should $z$ be?

The Laplace approximation for logistic regression defines $z= \mathbf{w}_{MAP}$ when performing the second order Taylor expansion. Performing this calculation for the posterior looks a little ugly but is surprisingly straightforward. There are essentially 2 observations needed and the rest is bookkeeping.

We will give the steps first then explain them afterwards:

<div class="math">
\begin{alignat*}{1}
p(\mathbf{w} | X, \mathbf{y}) &=\frac{\exp\left\{f(\mathbf{w})\right\}}{\int \exp\left\{f(\mathbf{w})\right\} \, d\mathbf{w}} \tag{L0} \\[5pt]
& \approx \frac{\exp\left\{f(z)+(\mathbf{w}-z)^{T} \nabla f(z)+\frac{1}{2}(\mathbf{w}-z)^{T}\left(\nabla^{2} f(z)\right)(\mathbf{w}-z)\right\}} {\int \exp\left\{f(z)+(\mathbf{w}-z)^{T} \nabla f(z)+\frac{1}{2}(\mathbf{w}-z)^{T}\left(\nabla^{2} f(z)\right)(\mathbf{w}-z)\right\} \, d\mathbf{w}} \tag{L1}\\[5pt]

& \approx \frac{\exp\left\{-\frac{1}{2}\left(\mathbf{w}-\mathbf{w}_{MAP}\right)^{T}\left(-\nabla^{2} \ln p\left(\mathbf{y}, \mathbf{w}_{MAP}|X\right)\right)\left(\mathbf{w}-\mathbf{w}_{MAP}\right) \right\}  }{\int
\exp\left\{-\frac{1}{2}\left(\mathbf{w}-\mathbf{w}_{MAP}\right)^{T}\left(-\nabla^{2} \ln p\left(\mathbf{y}, \mathbf{w}_{MAP}|X \right)\right)\left(\mathbf{w}-\mathbf{w}_{MAP}\right) \right\} \, d\mathbf{w}}. \tag{L2}

\end{alignat*}
</div>

###### Comments on the steps
<div class="bullet"> 
<li> $\text{(L0)}$: this is by definition $f(\mathbf{w})$ </li>
<li> $\text{(L0)}$ to $\text{(L1)}$: this uses the second-order Taylor expansion for $f(\mathbf{w})$ around $z$. </li>
<li> $\text{(L1)}$ to $\text{(L2)}$: this uses 2 observations:</li>
  <ul>
  <li> $f(z)$ does not depend on $\mathbf{w}$ and so can be viewed as a constant in both the numerator and denominator which cancels out. </li>
  <li> $\nabla f(z) = 0$ when evaluated at $z = \mathbf{w}\_{MAP}$ by definition of $\mathbf{w}\_{MAP}$ being a maximum. </li>
  </ul>
</div>
<br>
We are thus only left with the following term to worry about in the both the numerator and denominator:

<div class="math">
\begin{align*}
\frac{1}{2}(\mathbf{w}-z)^{T}\left(\nabla^{2} f(z)\right)(\mathbf{w}-z)
\end{align*}
</div>

which we can rewrite slightly by setting $z = \mathbf{w}\_{MAP}$, using the definition of $f(z)$ and pulling a negative out the front to match the form of a Gaussian, giving:

<div class="math">
\begin{align*}
-\frac{1}{2}\left(\mathbf{w}-\mathbf{w}_{MAP}\right)^{T}\left(-\nabla^{2} \ln p\left(\mathbf{y}, \mathbf{w}_{MAP}|X\right)\right)\left(\mathbf{w}-\mathbf{w}_{MAP}\right)
\end{align*}
</div>

###### Recognising the solution is a Gaussian
Looking at $\text{(L2)}$ we see this is in the form of a multivariate Gaussian with:

<div class="math">
\begin{align*}
\mu=\mathbf{w}_{MAP}, \quad \Sigma=\left(-\nabla^{2} \ln p\left(\mathbf{y}, \mathbf{w}_{MAP} | X\right)\right)^{-1}
\end{align*}
</div>

where we still need to be able to calculate the derivative in $\Sigma$. This derivative is the second derivative (Hessian) of the log joint likelihood (details given below [here](#deriv_log_like)) and results in:

<div class="math">
\begin{align*}
\nabla^{2} \ln p\left(\mathbf{y}, \mathbf{w}_{MAP} | X \right)=-\lambda I-\sum_{i=1}^{n} \sigma\left(y_{i} \cdot  \mathbf{w}_{MAP}\right)\left(1-\sigma\left(y_{i} \cdot \mathbf{w}_{MAP}\right)\right) \mathbf{x}_i \mathbf{x}_i^T
\end{align*}
</div>

which means we now we have expressions for both $\mu$ and $\Sigma$ in the Gaussian approximation to the posterior:

<div class="math">
\begin{align*}
p(\mathbf{w} | X, \mathbf{y}) \approx \mathcal{N}(\mu, \Sigma)
\end{align*}
</div>

<a name="deriv_log_like"></a>

##### Second derivative of the log joint likelihood

To compute $\Sigma$ we need to calculate:

<div class="math">
\begin{align*}
\nabla_{\mathbf{w}}^2 \ln p\left(\mathbf{y}, \mathbf{w} | X \right)
\end{align*}
</div>

at the point $\mathbf{w}=\mathbf{w}\_{MAP}$. Recall also that the prior is a Gaussian $\mathbf{w} \sim \mathcal{N}\left(0, \lambda^{-1} I\right)$.

We provide the breakdown of this calculation for reference:

<div class="math">
\begin{alignat*}{1}

\nabla_{\mathbf{w}}^2 \ln p\left(\mathbf{y}, \mathbf{w} | X\right)
&= \nabla_{\mathbf{w}}^2 \ln \Big\{ \overbrace{p(\mathbf{w})}^\text{prior} \,  \overbrace{\underbrace{\prod_{i=1}^{n} \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right)}_\text{$p(\mathbf{y} | \mathbf{w}, X )$}}^\text{likelihood} \Big\} \\[5pt]
&= \nabla_{\mathbf{w}}^2 \ln \Big\{ A \exp\{ \frac{-\mathbf{w}^2 \lambda I}{2} \} \prod_{i=1}^{n} \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right) \Big\} \hspace{1cm} &\text{defn. of prior} \\[5pt]
&= \nabla_{\mathbf{w}}^2 \left[ \ln  A + \frac{-\mathbf{w}^2 \lambda I}{2} + \sum_{i=1}^{n} \ln \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right) \right] &\text{rule of logs}  \\[5pt]
&= -\lambda I - \nabla_{\mathbf{w}}^2 \left[ \sum_{i=1}^{n} \ln \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right) \right] &\text{differentiate w.r.t. $\mathbf{w}$} \\[5pt]
&= -\lambda I - \sum_{i=1}^{n} \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right) \left(1 - \sigma_{i} \left(y_{i} \cdot \mathbf{w}\right) \right) \mathbf{x}_i \mathbf{x}_i^T &\text{by deriv. of sigmoid}
\end{alignat*}
</div>

where the last step follows from the definition of the derivative of the sigmoid function and isn't hard to show (we don't show it here as things are already getting a little lengthier than intended). $A$ in the above is the normalizing constant for the Gaussian prior and it disappears when differentiated so we chose to simplify the notation and not include the full form.

The above is the form that will be used to calculate $\Sigma$ as required at the point $\mathbf{w}=\mathbf{w}\_{MAP}$.

<a name="mercer"></a>
##### Mercer's theorem

Mercer's theorem assures us that by computing $\kappa(\mathbf{x}_i, \mathbf{x}_j)$ for any symmetric positive-definite choice of the kernel, $\kappa$, it's possible to find a transformation $\phi : \mathbb{R}^d \rightarrow \mathbb{R}^D$ such that:

<div class="math">
\begin{align*}
\kappa(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j).
\end{align*}
</div>

The reason this is of interest is that as opposed to thinking in terms of explicit expressions for $\phi$ (which may be very complicated) we can instead postulate a kernel function $\kappa$ that captures the prior beliefs we have about the correlation structure. We can actually do this without even needing to know what the equivalent feature mapping $\phi$ was that gave rise to it - Mercer's theorem guarantees such a $\phi$ exists.

This is handy as lots of machine learning algorithms can now be extended by replacing the dot product between data-points with a kernel function.

For a hands on practical example of why using $\kappa$ is easier than $\phi$ see my post [here.](/../posts/2018/10/19/The-kernel-trick)
<hr class="with-margin">
