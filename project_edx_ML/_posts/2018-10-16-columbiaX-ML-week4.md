---
layout: post
title: ColumbiaX - ML - week 4
date: 2018-10-16
use_math: true
image: "knn.jpg"
comments: true
tags: [classification, bayes classifier, kNN, perceptron]
---
This week introduces classification with the $k$-nearest neighbours algorithm before introducing the Bayes classifier as the optimal classifier with LDA and QDA as approximations. We then move on to general linear classifiers by discussing the perceptron, an iterative algorithm and the precursor to modern neural networks.

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

This week starts with the supervised learning problem of classification.

[$k$-nearest neighbours](#knn) is introduced as an intuitive and simple classifier before we break to talk about the theoretically optimal classifier, the [Bayes classifier](#bayes_class_app). Here, in particular, we spend time providing an example to illustrate why the Bayes classifier is optimal.

However, whilst the Bayes classifier itself is not of any practical use, it provides a framework for an approach to approximating the optimal classifier. In this context we look at an example of a [generative classifier](#gen_classifier) which is then tied to both the [LDA and QDA](#lda_qda) models. The [naive Bayes classifier](#naive_bayes) is also introduced along the way.

Both LDA and QDA have closed form solutions and so next we move to introduce general linear classifiers and the first such linear classifier we meet is the [perceptron](#perceptron) algorithm.

The [appendix](#hyperplanes) provides some results and explanation of the geometry of hyperplanes for linear classifiers.

<a name="approaches"></a>
<hr class="with-margin">
<h4 class="header" id="approaches">Approaches to classification</h4>


<a name="bayes_rule_class"></a>
<blockquote class="tip">
<strong>Recall</strong>
<br>
It's worth recalling Bayes' rule for classification, where the goal is to assign each observation to one of $k$ classes:

<div class="math">
\begin{align*}
\color{#e5c07b}{p\left(c_{k} \mid \mathbf{x}\right)}=\frac{ \color{#e06c75}{p\left(\mathbf{x} \mid c_{k}\right)} \color{#61afef}{p\left(c_{k}\right)}}{\color{#98c379}{p(\mathbf{x})}} \tag{BR}
\end{align*}
</div>

with terms: <span style="color:#e5c07b">class posterior</span>, <span style="color:#e06c75">data likelihood given class</span>, <span style="color:#61afef">class prior</span> and <span style="color:#98c379">evidence</span>.
<br>
<br>
Note $p(\mathbf{x})$ can be calculated by summing out the classes:

<div class="math">
\begin{align*}
p(\mathbf{x})=\sum_{k} p\left(\mathbf{x} \mid c_{k}\right) p\left(c_{k}\right).
\end{align*}
</div>
</blockquote>

Before we dive into classification algorithms, it's important to see the lay of the land. [PRML](#prml) sets the scene and provides an overview of the main approaches to classification, essentially defining 3 approaches, decreasing in complexity:

##### 1. Posterior via generative model

This approach models the posterior by first determining the class-conditional densities, $p\left(\mathbf{x} \mid c_{k}\right)$, for each $c_k$ individually. We also separately infer the prior class probabilities, $p\left(c_{k}\right)$. We then use Bayes' rule to solve for the posterior $p\left(c_{k} \mid \mathbf{x}\right)$\*.

Examples: LDA, QDA, Naive Bayes.

\*for some algorithms which use a generative model it's not always necessary to calculate $p(\mathbf{x})$, and hence calculate the full posterior, in order to make a classification decision. We discuss this briefly in the [appendix](#inference_decision).

##### 2. Posterior via discriminative model

The discriminative approach models the posterior, $p\left(c_{k} \mid \mathbf{x}\right)$, directly by specifying a model that outputs class probabilities.

Examples: logistic regression.

##### 3. Discriminant function

This approach finds a function $f(\mathbf{x})$ that directly maps each input $\mathbf{x}$ onto a class label. For instance, we might fit a hyperplane between two classes and declare everything to one side of the hyperplane to be in one class and everything on the other side to be in the other class. In this approach we do not model probabilities.

Examples: SVMs, perceptron algorithm.

##### A note on the above

It is not particularly common to distinguish between approaches 2 and 3 above and some references simply bundle them together. Further, the first two approaches really define the *inference* stage of classification and we then need to use the class probabilities to reach a *decision* about which class to assign a data-point to. Approach 3 directly reaches a decision.

This is discussed further in the [appendix](#inference_decision). 

<a name="knn"></a>
<hr class="with-margin">
<h4 class="header" id="knn">k-nearest neighbors (k-NN)</h4>

##### Introduction
The $k$-nearest neighbours classifier is one of the simplest and most intuitive places to start with classification algorithms. It is a non-parametric model meaning it doesn't make any assumptions about the underlying data distribution, except in assuming it cannot be defined by a finite set of parameters. It is also what is called a *lazy learning* method in that the generalization of the training data is deferred until the query for a test data-point is made. In this way there is no explicit training stage.  

##### $k$-nearest neighbours algorithm

<blockquote class="algo">
<hr class="small-margin">
<strong>Algorithm: $k$-nearest neighbours</strong>
<hr class="small-margin">
Given data $(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)$ with $\mathbf{x}_i \in \mathbb{R}^d$ and $y_i$ a class label.
<br>
<br>
For a data-point $\mathbf{x}_0$ not in the training data:
<br>
<br>
1. Find the $k$ nearest data-points in the training set to $\mathbf{x}_0$ based on a distance metric.
<br>
2. Assign $\mathbf{x}_0$ the label of the most common class amongst its $k$ nearest neighbours.
</blockquote>

So the point $\mathbf{x}_0$ will be assigned to the class based on a vote amongst the $k$ data-points it is most similar to in the training data.

##### Measuring distance

There are many options open to us in terms of how to measure distance. The crux of the $k$-NN algorithm is to be able to quantify in some way, how similar data-points are to each other. A common metric used for two points $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$ is the Euclidean distance:

<div class="math">
\begin{align*}
||\mathbf{a} -\mathbf{b}||_{2} = \left(\sum_{j=1}^{d} \left(\mathbf{a}_j -\mathbf{b}_j \right)^{2} \right)^{\frac{1}{2}}
\end{align*}
</div>

However, whilst this is an intuitive and appealing metric to use it is not particularly suitable for high-dimensional data as we discuss briefly [shortly](#knn_high_dim).

##### How do we choose $k$?

Typically via cross-validation.

The parameter $k$ essentially controls the model complexity and as such dictates the bias-variance tradeoff for the $k$-NN algorithm. A small $k$ will lead to more complex decision boundaries as few neighbours are considered when making a classification decision. This makes the model very flexible and able to closely fit the shape of the dataset.

The chart below (Model 3) shows how more complex decision boundaries can lead to better training error but worse test error - this is a classic case of overfitting (low bias, high variance) which would be achieved with small $k$. Alternatively, too simple a model can lead to high bias and low variance - this is underfitting and both the training and test error are poor. Model 2 shows a compromise that minimizes test error, which is ultimately what we care about.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/kNN_bv_tradeoff.png" alt="Image" width="600" height="800" />
</p>
<em class="figure">Varying model complexity from large $k$ (right) to small $k$ (left)
<br> [Image credit](https://cambridgecoding.wordpress.com/2016/03/24/misleading-modelling-overfitting-cross-validation-and-the-bias-variance-trade-off/)</em>

<a name="knn_high_dim"></a>
##### A note on $k$-NN with high-dimensional data

It might be expected that given enough training data $k$-NN can perform well, even for data with many dimensions. However, this is not the case as the concept of locality and hence neighbours breaks down when $d$ becomes large and algorithms reliant on concepts of distance suffer from the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality#Distance_functions).

Further discussion of this phenemonem is provided in [ESL \[2.5\]](#esl).

We now move onto discussing the theoretically best classifier, the Bayes classifier.

<a name="bayes_class_app"></a>
<hr class="with-margin">
<h4 class="header" id="bayes_class_app">Bayes classifier</h4>

<blockquote class="tip">
<strong>TLDR:</strong> the Bayes classifier is the theoretically best possible classifier if we knew the true underlying distribution, $\mathcal{P}$, from which the data is generated, independent of any other data-point:

<div class="math">
\begin{align*}
(X, Y) \stackrel{iid}{\sim} \mathcal{P}
\end{align*}
</div>

It is the classifier that has the smallest expected prediction error amongst all classifiers, or equivalently, the best prediction accuracy.
<br>
<br>
It provides a framework for modelling classifiers as well as theoretical grounding for an important, and intuitive, decision heuristic commonly used across machine learning: assigning an example to the class with highest posterior probability.
<br>
<br>
<strong>Remember:</strong> if we had access to the true data distribution, forming an optimal classifier would be trivial.
</blockquote>

##### Introduction

The Bayes classifier is a theoretical classifier that permits examination of what an optimal classifier might look like. Perhaps unsurprisingly, for a theoretically optimal classifier, we practically cannot use it as the assumption it makes (knowledge of $\mathcal{P}$) does not hold in reality.

Nevertheless, the Bayes classifier shows that with knowledge of $\mathcal{P}$ the optimal decision to minimise the expected prediction error is to simply assign an example to the class with the highest posterior probability. This result provides justification for the effort in machine learning that is put into finding ways to estimate the class posterior probabilities in order to try to approximate this unattainable gold standard.

Given $\mathcal{P}$, the Bayes classifier is a very simple classifier that produces the lowest possible test error rate, called the Bayes error rate.

##### Statement of the Bayes classifier

The Bayes classifier, $f^{\star}$, has the property that it minimizes the expected prediction error, or equivalently by Bayes' rule, maximizes the class posterior probability:

<div class="math">
\begin{alignat*}{1}
f^{\star}(x) &= \arg \max_{y \in \mathcal{Y}} p(Y=y \mid X=x)  \\[5pt]
&= \arg \max_{y \in \mathcal{Y}} \underbrace{p(Y=y)}_{\text {class prior}} \, \underbrace{p(X=x \mid Y=y)}_{\text { data likelihood } \mid \text {class}} \hspace{2cm} \text{(by Bayes' rule)} \tag{BC}
\end{alignat*}
</div>

for discrete random variables $Y$ and $X$. If $X$ is continuous, we use instead $p(\mathbf{x} \mid Y=y)$ as a class conditional density function.

We illustrate the above shortly with a concrete example.

<blockquote class="tip">
<strong>Note on terminology:</strong> by classifier we mean any function that takes in a particular assignment of $X$ (i.e. $X=x$ in the discrete case) and outputs a class label $y \in \mathcal{Y}$. Such a classifier could be any type of (potentially complicated) model. We will use the terms classifier and decision rule interchangeably in the discussion that follows.
<br>
<br>
Note that the classifiers are deterministic and given the same input example $x$ will always output the same class prediction $y$.
</blockquote>

##### Extension to plug-in classifiers
In practice we don't know the true underlying data distribution, $\mathcal{P}$, and hence we don't know the prior, $p(Y=y)$, or the likelihood, $p(X=x \mid Y=y)$. Instead all we have is some training data which is drawn from $\mathcal{P}$.

The approach used in practice is thus to use this data to approximate both the prior and the likelihood. Doing this is sometimes called using a plug-in classifier, as per the [generative classifier](#gen_classifier) discussed in the next section.

Of course, once we use an approximation we no longer have an optimal classifier.

##### Intuitively understanding why the Bayes classifier is optimal

Proofs on the optimality of the Bayes classifier can often obscure what is a highly intuitive result. We now walk through an example constructed to hopefully convey this fact. The example assumes knowledge of $\mathcal{P}$.

Suppose we have data from 2 classes with prior probabilities:

<div class="math">
\begin{align*}
p(y_1) = 0.7, \, \, \, p(y_2) = 0.3
\end{align*}
</div>

and $X \in \\{1,2,3,4,5 \\}$ is a discrete random variable, a particular instance of which is called $x$.

Suppose we also have the distribution of the likelihood, $p(X=x \mid Y=y)$ as shown in Table 1 below:

<p align="center">
    <img src="/assets/img/bayes_ex1.png" alt="Image" width="500" height="80" />
</p>
<em class="figure">Table 1: data likelihood, $p(X=x \mid Y=y)$, each row sums to 1</em>
<hr class="small-margin">

The blue cells in Tables 1 and 2 are the result of an arbitrary decision rule from a classifier and highlight the classification decision this (made-up) classifier would make for each $x$. Such a decision for every $x$ amounts to selecting a row for each column in the table.

Given the class priors and data likelihood we can also formulate the joint probability using:

<div class="math">
\begin{align*}
p(X=x, Y=y) = p(X =x \mid Y=y)p(Y=y)
\end{align*}
</div>

by multiplying the entries in each row by the prior class probability. This joint probability is shown in Table 2 below:

<p align="center">
    <img src="/assets/img/bayes_ex2.png" alt="Image" width="500" height="80" />
</p>
<em class="figure">Table 2: joint probability distribution, $p(X=x, Y=y)$, this must sum to 1.</em>
<hr class="small-margin">

Now, given the arbitrary classifier (per the blue predictions) we can think about the average error of this classifier. In the case of 2 classes there are 2 sources of error:
<div class="bullet"> 
<ol> 1. Choosing class $y_1$ when $y_2$ was the correct label, happens with probability equal to the sum of non-highlighted entries in $y_2$ row in Table 2. </ol>
<ol>2. Choosing class $y_2$ when $y_1$ was the correct label, happens with probability equal to the sum of non-highlighted entries in $y_1$ row in Table 2.</ol>
</div>
<br>
To see the above, consider that if every time the classifier encounters $x=1$ it predicts class label $y_2$ it will incur an error equal to:

<div class="math">
\begin{align*}
p(x=1 \mid y=1)p(y = 1) = p(x=1, y=1).
\end{align*}
</div>

Looking at Table 2 this means that the classifier will be wrong 14% of time from the fact that when $x=1$ it always predicts $y=2$ and we know from the true underlying joint distribution (this is what Table 2 is, this is what knowledge of $\mathcal{P}$ allows us to calculate!) that the pair $(x=1, y=1)$ will occur in 14% of the cases.

This is the same for $(x=2, y=2)$ which happens with probability ~3% and similarly for all non-highlighted cells in Table 2.

The total error is thus the sum of the joint probability for every data pair that the classifier predicts doesn't happen, this is a sum over both $x$ and $y$ given by the the non-highlighted cells in Table 2. The expected prediction error based on the classifier's predictions per the blue cells is thus $0.37$.

Thus, given some arbitrary classifier (as per the blue cells) the total error it will make on average depends on the true underlying probability distribution. This error happens in part because $p(Y \mid X)$ is [random](#whats_random).

##### Enter the Bayes classifier

What is the optimal classifier/decision rule to apply?

Well it turns out the expected prediction error is minimised when we choose to assign each $x$ to the class with highest joint probability (this is proportional to the posterior probability) in Table 2. This follows instinctively from the above discussion about the total error being the sum of the joint probability for every data pair that the classifier predicts doesn't happen. In other words, we simply make a prediction to minimize the sum of the non-highlighted cells, this happens when we predict for each $x$ the label with the highest joint probability of occurring.

This is the idea behind the Bayes classifier and the decision based on this is shown in Table 3.

<p align="center">
    <img src="/assets/img/bayes_ex3.png" alt="Image" width="500" height="80" />
</p>
<em class="figure">Table 3: Decision using highest joint probability.</em>
<hr class="small-margin">

This gives an expected prediction error of $0.25$, and no other classifier can beat this Bayes classifier, although it isn't guaranteed to be unique. Of course, we don't usually ever know $\mathcal{P}$ and so we approximate the prior and likelihood in practice - this leads to sub-optimal classifiers.

<a name="whats_random"></a>
<blockquote class="tip">
<strong>A note on what is random:</strong> it's important to note that the decision rule/classifier itself is not random. Given an $x$ it will always predict the same class label (predictions above in blue). It is the data itself that is assumed jointly randomly generated from some underlying distribution $\mathcal{P}$. An implication of this is that $X \mid Y$ and $Y \mid X$ are both random.
<br>
<br>
An example is for spam email. Consider an email string "Hey Bob, fancy checking out the new credit card deal?"
<br>
<br>
This exact email string might be spam 95% of the time but not spam in the other 5% of the time. Thus $Y \mid X$ is random.
</blockquote>

##### Final subtlety on Bayes classifier

The Bayes optimal classifier is the best classifier (smallest expected prediction error) amongst all classifiers given knowledge of $\mathcal{P}$.

However optimal Bayes classification is sometimes introduced differently, without assuming knowledge of $\mathcal{P}$ from which the data $\mathcal{D}$ was generated.

We call $\mathcal{F}$ the set of all possible classifiers and in this case the best classification decision given data, $\mathcal{D}$, is:

<div class="math">
\begin{align*}
\underset{y \in \mathcal{Y}}{\arg \max } \color{#e06c75}{\sum_{f_{i} \in \mathcal{F}}} \color{#61afef}{p\left(y \mid f_{i}\right)} \color{#98c379}{ p\left(f_{i} \mid \mathcal{D}\right)}.
\end{align*}
</div>

In other words, we assign the label based on the highest probability as computed by the <span style="color:#61afef">prediction from a given classifier</span> weighted by the <span style="color:#98c379">posterior probability of that classifier given the data</span> <span style="color:#e06c75">summing over all possible classifiers</span>.

Given that classifiers are things like decision trees, NNs etc... this is clearly infeasible practically but is theoretically optimal given infinite compute power.

This is a little confusing terminology wise as the above is now not a single classifier and so is more appropriately referred to as the Bayes optimal classification.

Further discussion can be found [here](https://www.youtube.com/watch?v=mjdJSReJ4e4&list=PLTPQEx-31JXillYa8apaBkuXLWKzOKzY8&index=6).

<a name="gen_classifier"></a>
<hr class="with-margin">
<h4 class="header" id="gen_classifier">Example: a generative classifier</h4>

##### Introduction
We now consider an example of a plug-in [generative classifier](#approaches) where we approximate both the prior and likelihood. Here, $X \in \mathbb{R}^d$ is now a continuous random variable with binary target $Y$ such that $Y \in \\{0,1\\} = \mathcal{Y} $ and an instance of $X$ is a vector $\mathbf{x} \in \mathbb{R}^d$.

Per the justification provided by the [Bayes classifier](#bayes_class_app) we assign an instance to the class which maximizes the numerator in Bayes' rule:

<a name="gen_classifier_opt"></a>

<div class="math">
\begin{align*}
\arg \max_{y \in \mathcal{Y}} \underbrace{p(Y=y)}_{\text {class prior}} \, \underbrace{p(\mathbf{x} \mid Y=y).}_{\text { data likelihood } |=\mid \text {class}} \tag{0}
\end{align*}
</div>

In this case we are not obtaining a full posterior but simply using the fact that:

<div class="math">
\begin{align*}
\arg \max_{y \in \mathcal{Y}} \underbrace{p(Y=y \mid \mathbf{x})}_\text{class posterior} = \arg \max_{y \in \mathcal{Y}} \underbrace{p(Y=y)}_{\text {class prior}} \, \underbrace{p(\mathbf{x} \mid Y=y)}_{\text { data likelihood } \mid \text {class}} \tag{1}
\end{align*}
</div>

and the above is valid as the denominator in [Bayes' rule](#bayes_rule_class) does not depend on the class lable, here $y$.

##### Gaussian class conditional densities

<blockquote class="tip">
<strong>TLDR:</strong> model the data from each class as a multivariate Gaussian distribution with parameters estimated from the data.
</blockquote>

We approximate the prior, $p(Y = y)$, and the likelihood, $p(\mathbf{x} \| Y=y)$ as:

<div class="math">
\begin{alignat*}{1}
\hat{\pi}_{y} = p(Y = y) &=  \underbrace{\frac{1}{n} \sum_{i=1}^{n} 1_{\left(y_{i}=y\right)}}_\text{class fraction} \\[5pt]
p(\mathbf{x} \mid Y=y) &= \underbrace{\mathcal{N} \left(\mathbf{x} \mid \boldsymbol{\mu}_{y}, \Sigma_{y}\right)}_\text{class conditional Gaussian}
\end{alignat*}
</div>

where the maximum likelihood estimates for $\boldsymbol{\mu}\_y, \Sigma_{y}$ are given by:

<div class="math">
\begin{alignat*}{1}
\hat{\boldsymbol{\mu}}_{y} &= \underbrace{\frac{1}{n_{y}} \sum_{i=1}^{n} 1_{\left(y_{i}=y\right)} \mathbf{x}_{i}}_\text{sample mean of class $y$} \\[5pt]
\hat{\Sigma}_{y} &= \underbrace{\frac{1}{n_{y}} \sum_{i=1}^{n} 1_{\left(y_{i}=y\right)} \left(\mathbf{x}_{i}-\hat{\boldsymbol{\mu}}_{y}\right)\left(\mathbf{x}_{i}-\hat{\boldsymbol{\mu}}_{y}\right)^{T}}_\text{sample covariance of class $y$}.
\end{alignat*}
</div>

In other words for each class in the training data we calculate its mean and covariance and use these to estimate the class conditional Gaussians. The prior probability of a class is just set to be the fraction of training examples in each class. It's important to note that all calculations are able to be done in closed form in this case.

Dimensions for reference with respect to the above:
<div class="bullet"> 
<li> $\mathbf{x}_i$: $d \times 1$ </li>
<li> $(\mathbf{x}_i-\hat{\boldsymbol{\mu}}_y)$: $d \times 1$ </li>
<li> $\hat{\boldsymbol{\mu}}_y$: $d \times 1$ </li>
<li> $\hat{\Sigma}_y$: $d \times d$ </li>
</div>
<br>
##### Plug-in classifier decision

Using the above assumptions and dropping any terms not dependent on $y$ we can write Gaussian class conditional decision as:

<div class="math">
\begin{align*}
\arg \max_{y \in \mathcal{Y}} \hat{\pi}_{y}\left|\hat{\Sigma}_{y}\right|^{-\frac{1}{2}} \exp \left\{-\frac{1}{2}\left(\mathbf{x}-\hat{\boldsymbol{\mu}}_{y}\right)^{T} \hat{\Sigma}_{y}^{-1}\left(\mathbf{x} -\hat{\boldsymbol{\mu}}_{y}\right)\right\} \tag{2}
\end{align*}
</div>

whereby an instance is assigned to the class $y$ that maximizes (2).

We will return to the discussion of the above classifier in the next section on LDA and QDA, however we segue briefly to introduce another popular classifier.

<a name="naive_bayes"></a>
<hr class="with-margin">
<h4 class="header" id="naive_bayes">Naive Bayes classifier</h4>

We very briefly mention the naive Bayes classifier now which makes the assumption that the covariates of $X$ are conditionally independent given $y$. In other words, there is no correlation between the features given the class. This allows us to write the class conditional densities as a product of one dimensional densities:

<div class="math">
\begin{align*}
p(X=x \mid Y=y)=\prod_{j=1}^{d} p_{j}(x_j \mid Y=y) \tag{3}
\end{align*}
</div>

where $x_j$ denotes the $j$th feature of $X$.

Whilst we do not expect these assumptions to hold the NBC often performs well in practice, one reason being that it is a simple model with few parameters and so is reasonably robust to over-fitting. NBC is also highly scalable requiring a number of parameters which is linear in the number of features in $X$.

The classic example of NBC performing well is in the context of [spam filtering](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering).

<a name="lda_qda"></a>
<hr class="with-margin">
<h4 class="header" id="linear_class">LDA and QDA</h4>

<blockquote class="tip">
<strong>TLDR:</strong> LDA assumes that the data for each class come from a Gaussian distribution with a class-specific mean vector and a shared covariance matrix across all classes, whereas QDA computes a per class covariance matrix.
</blockquote>

##### Introduction

In this section we look at two common algorithms for classification called linear discriminant analysis (LDA) and quadratic discriminant analysis (QDA), both of which are generative classifiers. Recall that in the example of the [generative classifier](#gen_classifier) above, we assumed class conditional Gaussian distributions with mean and covariance matrices estimated from the data. 

In particular:
<div class="bullet"> 
<li> if we instead use the same covariance matrix across all classes the resulting decision boundary is linear and this model is called linear discriminant analysis (LDA). </li>
<li> estimating a covariance matrix per class gives a decision boundary that is quadratic and the model is called quadratic discriminant analysis (QDA).</li>
</div>
<br>
<blockquote class="tip">
<strong>Sidebar on log odds for classification:</strong> recall that for the generative classifier we assigned an instance to the class with the highest posterior probability where the justification for this came from the Bayes classifier.
<br>
<br>
In the case of binary classification this is equivalent to declaring an observation to be in class 1 if:

<div class="math">
\begin{alignat*}{1}
\frac{p(\mathbf{x} \mid y=1) p(y=1)}{p(\mathbf{x} \mid y=0) p(y=0)} &> 1 \iff \\[5pt]
\underbrace{\ln \frac{p(\mathbf{x} \mid y=1) p(y=1)}{p(\mathbf{x} \mid y=0) p(y=0)}}_\text{log odds} &> 0. \tag{4}
\end{alignat*}
</div>

An advantage of using log odds to make a decision rule is that the denominator in Bayes’ rule cancels as it would appear on both the top and bottom of (4).

</blockquote>

##### Arriving at LDA

For the case where we have a shared covariance matrix across both classes, by the Gaussian likelihood assumption we made, calculating the log odds per equation (4) leads to:
<a name="lda_equation"></a>

<div class="math">
\begin{align*}
\begin{array}{l}{\underbrace{\ln \frac{\pi_{1}}{\pi_{0}}-\frac{1}{2}\left(\boldsymbol{\mu}_{1}+\boldsymbol{\mu}_{0}\right)^{T} \Sigma^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0}\right)}_{\text{ a constant, call $w_0$}}} {+\mathbf{x}^{T} \underbrace{\Sigma^{-1}\left(\boldsymbol{\mu}_{1}-\boldsymbol{\mu}_{0}\right)}_{\text { a vector, call } \mathbf{w}}}\end{array} \tag{5}
\end{align*}
</div>

which we can write in shorthand as $w_0 + \mathbf{x}^T\mathbf{w}$.

Based on equation (4) we can thus write the classification decision as:

<div class="math">
\begin{align*}
f(\mathbf{x}) = \operatorname{sign} (w_0 + \mathbf{x}^T\mathbf{w})
\end{align*}
</div>

where we are classifying an observation based on the resulting sign of $w_0 + \mathbf{x}^T\mathbf{w}$. It is important to remember that we can calculate all the terms in $w_0$ and $\mathbf{w}$ based on the maximum likelihood estimates we made for each $\pi_y, \mu_y$ as well as for $\Sigma$.

This is called linear discriminant analysis (LDA) and it has a linear decision boundary as seen from the fact that the decision rule is linear in $\mathbf{x}$.

##### Extending to QDA

Getting to QDA from LDA is by changing the assumption we make about $\Sigma$. If we now calculate a covariance per class for the likelihood, we have, $p(\mathbf{x} \| y) = \mathcal{N}\left(x \| \boldsymbol{\mu}\_y, \Sigma_{y} \right).$

By similar analysis as for LDA, working out the log odds from equation (4) gives:

<div class="math">
\begin{alignat*}{1}
\ln \frac{p(\mathbf{x} \mid y=1) p(y=1)}{p(\mathbf{x} \mid y=0) p(y=0)} &= \underbrace{\text{something complicated not depending on $\mathbf{x}$}}_\text{a constant} \\[5pt]

&+ {\underbrace{\mathbf{x}^{T}\left(\Sigma_{1}^{-1} \boldsymbol{\mu}_{1}-\Sigma_{0}^{-1} \boldsymbol{\mu}_{0}\right)}_{\text { a part that's linear in } \mathbf{x}}} \\[5pt]

&+ {\underbrace{\mathbf{x}^{T}\left(\Sigma_{0}^{-1} / 2-\Sigma_{1}^{-1} / 2\right) \mathbf{x}}_{\text { a part that's quadratic in } \mathbf{x}}}
\end{alignat*}
</div>

which is called quadratic discriminant analysis as it leads to quadratic shaped decision boundaries, but is linear in the weights. The resulting classifier for QDA, similar to LDA, can be written as:

<div class="math">
\begin{align*}
f(\mathbf{x}) = \operatorname{sign} (c + \mathbf{x}^T\mathbf{b} + \mathbf{x}^T A \mathbf{x})
\end{align*}
</div>

for a constant, $c$, a vector $\mathbf{b}$ and a matrix $A$. Again, all of $c$, $\mathbf{b}$ and $A$ are able to be computed in closed form from the maximum likelihood estimates we made for each $\pi_y, \mu_y$ and $\Sigma_y$.

An example of decision boundaries for LDA and QDA is shown below.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/lda_qda_sklearn.png" alt="Image" width="700" height="600" />
</p>
<em class="figure"> Chart plotting the covariance ellipsoids of each class and decision boundary learned by LDA and QDA. The ellipsoids display the double standard deviation for each class. With LDA, the standard deviation is the same for all the classes, while each class has its own standard deviation with QDA.<br>
Image credit: [sckit-learn](https://scikit-learn.org/stable/auto_examples/classification/plot_lda_qda.html#sphx-glr-auto-examples-classification-plot-lda-qda-py)</em>
<hr class="with-margin">

##### Discussion on LDA and QDA

Should we prefer LDA or QDA?

Analysis of this relates to the bias-variance trade-off. If $X$ has $d$ dimensions then estimating a single covariance matrix requires $d(d+1)/2$ parameters. Given QDA estimates a separate covariance matrix
for each class, this becomes $K d(d+1)/2$ parameters for $K$ classes. For high-dimensional data this can rapidly increase the number of parameters required to be estimated for which adequate data may not be available.

As the LDA model is linear in $\mathbf{x}$, it is a much less flexible classifier than QDA, and thus has lower variance. However, if the classes have significantly different covariance structures then LDA can suffer from high bias. In general QDA will tend to perform better if we have lots of data and the variance of the classifier is not a major concern, or if the assumption of a shared covariance matrix is clearly incorrect.

##### Linking LDA to a naive Bayes classifier

If in the QDA model we make the assumption that the covariance matrices are diagonal, then the inputs are assumed to be conditionally independent in each class, and the resulting classifier is equivalent to a [Gaussian Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes).

<a name="perceptron"></a>
<hr class="with-margin">
<h4 class="header" id="perceptron">The perceptron algorithm</h4>

<blockquote class="tip">
<strong>TLDR:</strong> assuming the data is linearly separable, use gradient descent to find a hyperplane defined by parameters $w_0$ and $\mathbf{w}$ in order to separate the data into classes.
</blockquote>

##### Introduction: a more general classifier

In the LDA model we classified the data based on:

<div class="math">
\begin{align*}
f(\mathbf{x}) = \operatorname{sign} (w_0 + \mathbf{x}^T\mathbf{w})
\end{align*}
</div>

where $w_0$ and $\mathbf{w}$ are given by explicit formulae. This can be too restrictive an assumption in general and so we now introduce a way to learn both $w_0$ and $\mathbf{w}$.

Learning $w_0$ and $\mathbf{w}$ amounts to learning a [hyperplane](https://en.wikipedia.org/wiki/Hyperplane) between the classes in the data. Details and intuition about hyperplanes are included in the [appendix](#hyperplanes).

The perceptron algorithm solves the binary classification problem by learning a hyperplane that separates the data and in order to do so it requires that the classes in the data be [linearly separable](https://en.wikipedia.org/wiki/Linear_separability). The loss function for the perceptron cannot be solved analytically and so gradient descent is used instead to learn $w_0$ and $\mathbf{w}$.

The perceptron is perhaps more (famously and) recently known as a core building block of modern neural networks.

<blockquote class="tip">
<strong>A point on notation:</strong> it is notationally simpler to absorb $w_0$ into the vector $\mathbf{w}$ by assuming we will add a column of 1s to the data matrix $X$. For the perceptron it is also more convenient to assume $y \in \{-1,1\}$ rather than  $y \in \{0, 1\}$ as we have previously done.
</blockquote>

##### The perceptron loss function

The above notational change means we can write a general linear classifier as:

<div class="math">
\begin{align*}
f(\mathbf{x}) = \operatorname{sign}\left(\mathbf{x}^{T}\mathbf{w}\right)
\end{align*}
</div>

and so we are predicting each example to be the sign of the dot product $\mathbf{x}_i^T\mathbf{w}$, with true label $y_i$.

Given the perceptron algorithm assumes the data is linearly separable, it tries to find a hyperplane that classifies every instance of the training data correctly. Such a loss function can be formulated as:

<div class="math">
\begin{align*}
\mathcal{L}=-\sum_{i=1}^{n} \overbrace{\left(y_{i} \cdot \mathbf{x}_{i}^{T} \mathbf{w}\right)
\underbrace{\mathcal{I} \left\{y_{i} \neq \operatorname{sign}\left(\mathbf{x}_{i}^{T} \mathbf{w}\right)\right\}.}_\text{$= \,1$ if misclassified, else $0$}}^\text{always negative as only counting misclassified}
\end{align*}
</div>

The perceptron loss function can be roughly explained as summing over some sense of distance from the hyperplane for all misclassified examples.

<blockquote class="tip">
<strong>Explaining the perceptron loss function in more detail</strong>
<br>
It is worth thinking about what the perceptron loss function means by examining the term $(y_{i} \cdot \mathbf{x}_{i}^{T} \mathbf{w})$:

<div class="math">
\begin{align*}
y_{i} \cdot \mathbf{x}_{i}^{T} \mathbf{w} \quad \text { is } \quad \left\{\begin{array}{l}{>0 \text { if } y_{i}=\operatorname{sign}\left(\mathbf{x}_{i}^{T} \mathbf{w}\right)} \\ {<0 \text { if } y_{i} \neq \operatorname{sign}\left(\mathbf{x}_{i}^{T} \mathbf{w}\right)}\end{array}\right.
\end{align*}
</div>

In words this is as follows:
<br>
<br>
If we classify a negative class example correctly then $\mathbf{x}_i^T \mathbf{w} < 0$ and $y_i$ is negative and so $y_i=\operatorname{sign}\left(\mathbf{x}_i^T \mathbf{w}\right)$. This means $(y_i \cdot \mathbf{x}_i^T \mathbf{w})$ is positive. Similarly, if we classify a positive class example correctly then $\mathbf{x}_i^T \mathbf{w} > 0$ and $y_i$ is positive and so $y_i = \operatorname{sign}\left(\mathbf{x}_i^T \mathbf{w}\right)$. This means $(y_i \cdot \mathbf{x}_i^T \mathbf{w})$ is positive.
<br>
<br>
So in both the cases where the perceptron is correct the term $(y_{i} \cdot \mathbf{x}_i^T \mathbf{w})$ is positive.
<br>
<br>
Similar analysis shows that when the perceptron is incorrect the term $(y_{i} \cdot \mathbf{x}_i^T \mathbf{w})$ is always negative. Thus, due to the indicator function, we sum over $(y_{i} \cdot \mathbf{x}_i^T \mathbf{w})$ only for every misclassified example. This sum of a bunch of negative terms is negative. We then negate the whole sum to make it positive and try to minimise it.
<br>
<br>
For more on hyperplanes and why $\mathbf{x}_{i}^{T} \mathbf{w}$ is measure of distance from the hyperplane see the <a class="reference external" href="{{page.url}}#hyperplanes">appendix.</a>
</blockquote>

##### The perceptron algorithm

Whilst we cannot solve the loss function analytically, we can differentiate it. The derivative for a misclassified observation is:

<div class="math">
\begin{align*}
\nabla_{\mathbf{w}} \mathcal{L}=-\sum_{i \in \mathcal{M}_{t}} y_{i} \mathbf{x}_{i}
\end{align*}
</div>

where $\mathcal{M}\_t$ is the set of misclassified instances. Thus the derivative for a single observation is $- y_{i} \mathbf{x}_{i}$. This is then used in the gradient descent update step.

Below is an example of the perceptron algorithm:
<a name="perceptron_algo"></a>
<blockquote class="algo">
<hr class="small-margin">
<strong>Algorithm: perceptron</strong>
<hr class="small-margin">
<strong>Input:</strong> training data $(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)$ and step size $\eta$.
<br>
<br>
1. <strong>Initialise:</strong> $\mathbf{w}^{(0)}$ to the zero vector
<br>
2. <strong>For step </strong>1, 2, ..., <strong>do:</strong>
<br>
&emsp; a) Search dataset for a misclassified example such that $y_{i} \neq \operatorname{sign}\left(\mathbf{x}_{i}^{T} \mathbf{w}\right)$
<br>
&emsp; b) <strong>If:</strong> such a $(\mathbf{x}_i, y_i)$ exists, randomly pick one and update:

<div class="math">
\begin{align*}
\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}+\eta y_{i} \mathbf{x}_{i}
\end{align*}
</div>
&emsp;&emsp; <strong>Else:</strong> return $\mathbf{w}^{(t)}$ as all examples are correctly classified.
</blockquote>

Below we show an image from [PRML](#prml) which illustrates the convergence of the perceptron learning algorithm, which is guaranteed for linearly separable data.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/perceptron.png" alt="Image" width="700" height="650" />
</p>
<em class="figure">The top left plot shows the initial parameter vector $\mathbf{w}$ shown as a black arrow together with the corresponding decision boundary (black line), in which the arrow points towards the decision region which classified as belonging to the red class. The data point circled in green is misclassified and so its feature vector is added to the current weight vector, giving the new decision boundary shown in the top right plot. The bottom left plot shows the next misclassified point to be considered, indicated by the green circle, and its feature vector is again added to the weight vector giving the decision boundary shown in the bottom right plot for which all data points are correctly classified.
<br> Image credit: [PRML](#prml)</em>
<hr class="with-margin">

A limitation of the perceptron algorithm is that even when the data set is linearly separable, there may be many solutions. The exact solution found will depend on the initialization of $\mathbf{w}^{(0)}$ and the order in which the algorithm sees the data-points. More concerning is that for data sets that are not linearly
separable, the perceptron learning algorithm will never converge.

We will later meet support vector machines, which is one such algorithm that addressees these concerns by modifying the objective function.

<hr class="with-margin">
<h4 class="header" id="references">References</h4>

<div class="bullet"> 
<li>
<a name="prml"></a>
Bishop, C. (2006). Chapters: 4.1; <a class="reference external" href="https://www.springer.com/gb/book/9780387310732">Pattern Recognition and Machine Learning</a>.</li>
<li>
<a name="esl"></a>
Hastie, T., R. Tibshirani, and J. Friedman (2001). Chapters: 2.3, 2.5, 4.5, 13.3 - 13.5;  <a class="reference external" href="http://web.stanford.edu/~hastie/ElemStatLearn/">The Elements of Statistical Learning</a>.</li>
<li>
<a name="edx_ml"></a>
edX, ColumbiaX, <a class="reference external" href="https://www.edx.org/course/machine-learning-1">Machine Learning</a>.</li>
<li>
Raschka, S, <a class="reference external" href="https://sebastianraschka.com/Articles/2014_python_lda.html">Linear Discriminant Analysis</a>.</li>
</div>

<a name="appendix"></a>
<hr class="with-margin">
<h4 class="header" id="appendix">Appendix: Q and A</h4>

<a name="inference_decision"></a>
##### Making a classification decision

If we simply wish to assign an instance to a class based on the highest posterior probability per the [Bayes classifier](#bayes_class_app) we needn't actually calculate the denominator in Bayes' rule as it doesn't depend on the class label $y$ and is common to all classes. We can instead simply choose the class such that the numerator in Bayes' rule is maximized:

<div class="math">
\begin{align*}
\arg \max_{y \in \mathcal{Y}} p(Y=y) \, p(X=x \mid Y=y). \tag{A1}
\end{align*}
</div>

###### Inference vs. decision making

We aren't bound to assign an instance to the class with the highest posterior probability and we now draw a distinction. It is more precise definitionally to call the inference stage the estimation of posterior probabilities and we could, in fact, after modelling the posterior probabilities, choose to implement a different decision rule to decide how to assign an observation to a class. This is the realm of decision theory and we only briefly mention here to highlight the difference between inference and decision making.

Fortunately and intuitively, in practice, the optimal decision step to minimize prediction error is generally as trivial as assigning an instance to the class with the highest posterior probability.

See [PRML [1.5]](#prml) for more details.

<a name="hyperplanes"></a>
##### Hyperplanes

Here we will discuss some of the geometric understanding of hyperplanes and how they are used for linear classifiers as well as deriving some of the facts from [PRML](#prml).

###### Relation to linear classifiers

Recall that we classify a point for a generic binary linear classifier according to:

<div class="math">
\begin{align*}
f(\mathbf{x})=\operatorname{sign}\left(\mathbf{x}^{T} \mathbf{w}+w_{0}\right)  \tag{A2}
\end{align*}
</div>
with $\mathbf{x}^T, \mathbf{w} \in \mathbb{R}^d$ and $w_0 \in \mathbb{R}$.

<blockquote class="tip">
<strong>Notation:</strong> to be consistent with <a class="reference external" href="{{page.url}}#prml">PRML</a> it will help to define:

<div class="math">
\begin{align*}
y(\mathbf{x}) = \mathbf{x}^{T} \mathbf{w}+w_{0}  \tag{A3}
\end{align*}
</div>

and it pays to not confuse $y$ with $f$, $f$ is just the sign of $y$.
</blockquote>

Given we assign a point to one class if $y(\mathbf{x}) \geq 0$ and the other class if $y(\mathbf{x}) \leq 0$ then $y(\mathbf{x}) = 0$ defines the decision boundary which in the 2-dimensional features space is a line.

To see this, consider:

<div class="math">
\begin{alignat*}{1}
\mathbf{x}^{T} \mathbf{w} + w_{0} &= \left(x_1 \, \, x_2\right) \left( \begin{array}{c}{w_1} \\ {w_2}\end{array}\right) + w_0 = 0 \\[5pt]
\Rightarrow x_2 &= -\frac{w_1x_1 + w_0}{w_2}  \tag{A4}
\end{alignat*}
</div>

which is the equation for a line for some arbitrary $\mathbf{w}$ and $w_{0}$. Thus as we vary $\mathbf{w}$ and $w_{0}$ we move the line around in the 2-dimensional feature space with $\mathbf{w}$ controlling its orientation and $w_0$ the displacement from the origin.

It's crucial to note that $\mathbf{w}$ is perpendicular to the hyperplane defined by $y(\mathbf{x}) = \mathbf{x}^T \mathbf{w}+w_0 = 0$. To see this consider any 2 points, $\mathbf{x}_A$ and $\mathbf{x}_B$, both of which lie on the decision boundary. Because $y(\mathbf{x}_A) = y(\mathbf{x}_A) = 0$, we have $\mathbf{w}^T (\mathbf{x}_A - \mathbf{x}_B) = 0$ and hence the vector $\mathbf{w}$ is orthogonal to every vector lying on the decision surface.

An illustration of the geometry of the problem is given below.

<hr class="with-margin">
<p align="center">
    <img src="/assets/img/hyperplane_bishop.png" alt="Image" width="550" height="500" />
</p>
<em class="figure">Geometry of a hyperplane as a linear discriminant function in 2-dimensions. The decision boundary is shown in red and is perpendicular to $\mathbf{w}$ with displacement from the origin controlled by $w_0$.
<br> Image credit: [PRML](#prml)</em>
<hr class="with-margin">

###### How far is a point $\mathbf{x}$ from the hyperplane?

The figure above shows the distance for a point $\mathbf{x}$ from the hyperplane as $\frac{y(\mathbf{x})}{\\|\mathbf{w}\\|}$, here we show why this is the case.

Let's call $r$ the perpendicular distance of a point from the hyperplane and consider an arbitrary point $\mathbf{x}$ in the 2-dimensional feature space. We call $\mathbf{x}_{\perp}$ its orthogonal projection onto the decision surface (i.e. the closest point on the hyperplane that is perpendicular to it).

Then we can state $\mathbf{x}$ as:

<div class="math">
\begin{align*}
\mathbf{x} = \color{#e06c75}{\mathbf{x}_{\perp}} + \color{#61afef}{r} \color{#98c379}{\frac{\mathbf{w}}{\|\mathbf{w}\|}} \tag{A5}
\end{align*}
</div>

where $\color{#98c379}{\frac{\mathbf{w}}{\\|\mathbf{w}\\|}}$ is the unit vector orthogonal to the hyperplane.

In words: any point $\mathbf{x}$ can be rewritten as <span style="color:#e06c75">a point on the hyperplane</span> plus <span style="color:#61afef">moving a distance</span> <span style="color:#98c379">away from the plane</span>. Recall here we define the point $\mathbf{x}_{\perp}$ such that to get to $\mathbf{x}$ we travel perpendicular from the hyperplane.

To prove $r = \frac{y(\mathbf{x})}{\\|\mathbf{w}\\|}$ we multiply both sides of (A5) by $\mathbf{w}^T$ and add $w_0$:

<div class="math">
\begin{alignat*}{1}
\underbrace{\mathbf{w}^T \mathbf{x} + w_0}_\text{$=\,y(\mathbf{x})$} &= \mathbf{w}^T \left(\mathbf{x}_{\perp} + r \frac{\mathbf{w}}{\|\mathbf{w}\|}\right) + w_0 \\[5pt]
&= \underbrace{\mathbf{w}^T \mathbf{x}_{\perp} + w_0}_\text{$= \, y(\mathbf{x}_{\perp})\, = \,0$} + r \mathbf{w}^T \frac{\mathbf{w}}{\|\mathbf{w}\|} \\[5pt]
&= r \frac{\|\mathbf{w}\|\|\mathbf{w}\|}{\|\mathbf{w}\|} \hspace{3cm} &\text{by defn of dot product} \\[5pt]
\Rightarrow r &= \frac{y(\mathbf{x})}{\|\mathbf{w}\|}
\end{alignat*}
</div>

as required.

###### How far is the hyperplane from the origin?

The figure above also shows the distance of the plane from the origin as $\frac{-w_0}{\\|\mathbf{w}\\|}$, here we show why this is the case.

To prove this call $\mathbf{x'}$ the point on the hyperplane that is closest to the origin. The distance of $\mathbf{x'}$ from the origin is the distance of interest and is thus equal to the length of the vector, $\\|\mathbf{x'}\\|$. We will come back to this fact shortly.

Given that $\mathbf{w}$ is orthogonal to the hyperplane we can write:

<div class="math">
\begin{align*}
\mathbf{x'} = \alpha \mathbf{w}
\end{align*}
</div>

for some scalar $\alpha$ which may be positive or negative. Then because $\mathbf{x'}$ is on the hyperplane we have:

<div class="math">
\begin{alignat*}{1}
\mathbf{w}^T \mathbf{x'} + w_0 &= 0 &\text{this is just $y(\mathbf{x'}) = 0$} \\[5pt]
\mathbf{w}^T \alpha \mathbf{w} + w_0 &= 0 \hspace{2cm} &\text{by defn of $\mathbf{x'}$} \\[5pt]
\Rightarrow \alpha &= -\frac{w_0}{ \mathbf{w}^T \mathbf{w}} \hspace{2cm} &\text{rearranging} \\[5pt]
\alpha &= -\frac{w_0}{ \|\mathbf{w}\|^2} \hspace{2cm} &\text{by defn of dot product} \\[5pt]
\end{alignat*}
</div>

where we have now found a value for $\alpha$ in terms of $\mathbf{w}$ and $w_0$ which we know.

Now, the distance of interest is:

<div class="math">
\begin{alignat*}{1}
\|\mathbf{x'}\| &= \| \alpha \mathbf{w} \| \\[5pt]
&= \alpha \| \mathbf{w} \| \\[5pt]
&= -\frac{w_0}{ \|\mathbf{w}\|^2} \| \mathbf{w} \| \hspace{2cm} &\text{plugging in value of $\alpha$} \\[5pt]
&= -\frac{w_0}{ \|\mathbf{w}\|}
\end{alignat*}
</div>
as required.

<hr class="with-margin">
