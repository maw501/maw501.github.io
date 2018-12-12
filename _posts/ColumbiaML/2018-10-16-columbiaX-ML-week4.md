---
layout: post
title: ColumbiaX - ML - week 4
date: 2018-10-16
use_math: true
image: "knn.jpg"
---
Week 4 started the topic of classification and touched on the k-NN classifier, Bayes classifier as the optimal classifier (the main idea of the week really) before moving onto linear classification and the perceptron algorithm.

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
<h4 class="header" id="intro">Week 4 (lectures 7 and 8): overview</h4>

I felt (again) that some of the intuition as to what we were trying to achieve was lost in a sea of dense notation so I hope I can simplify it a little below. The main ideas:

* **Classification**
  * The 'other' big domain supervised machine learning whereby we try to predict the class an example belongs to rather than a real value for it.
* **k-NN classifier**
  * The k nearest neighbours classifier is a simple idea where we simply predict a new data point's class by first computing the distance from this example to $k$ other examples and then assigning the majority label to our new example based on those of its $k$ nearest neighbours.
* **Bayes classifier**
  * This is a theoretically optimal classifier if we knew the distribution from which our data comes from. That is, with the assumption that $(X, Y) \stackrel{iid}{\sim} \Phi$ (for some distribution $\Phi$, which we don't know) then the optimal classifier $f^\*(x)$ assigns each data point to the class with the highest probability $P(Y=y \mid X=x)$ which we can think of as our posterior.
  * Of course, we don't actually know the underlying distribution $\Phi$ and so we need to approximate it. This is where Bayes rules comes into play by doing its usual trick of allowing us to turn the problem around and instead start to think about maximizing $P(Y=y) \, P(X=x \mid Y=y)$ instead of $P(Y=y \mid X=x)$.
* **Linear classifiers**
  * These are a set of classifiers where our model is linear in the learned parameters analogous to the regression setting. We can show that under certain assumptions the Bayes classifier is an instance of a linear classifier.
* **The perceptron algorithm**
  * This is an algorithm for binary classification that cannot be solved analytically (i.e. we can't differentiate the loss function and set to 0) so some form of learning algorithm (e.g. gradient descent) is required to learn the parameters. The perceptron assumes our data is linearly separable and so directly finds a hyperplane separating the data. The perceptron is more (famously and) recently known as a core building block of modern neural networks.

<hr class="with-margin">
<h4 class="header" id="big">Week 4 (lectures 7 and 8): the big picture</h4>

Week 4 really dug into the dirt with classifiers and, as is typical for people of a theoretical inclination, immediately started discussing an optimal classifier that we can't actually use in practice. However it did allow us to introduce the topic of **generative modelling** which actually has a lot advanced areas of research at the moment which are considered cutting edge. Here is my take on the main ideas in more detail.

##### In classification we wish to model $P(Y =y \mid X=x)$, Bayes rule allows us to flip this around and instead model $P(Y=y)$ and $P(X=x \mid Y=y)$

* The Bayes classifier represents the best we can do if we know the probability distribution from which our data comes
  * In this case it can be shown that in order to get the lowest misclassification error rate we can simply choose our predicted class as the class which has the highest probability, i.e. $\underset{y \, \in \, Y}{\operatorname{argmax}} P (Y=y \mid X=x)$ where we are assuming this probability comes from the true distribution $\Phi$.
* We don't know this distribution $\Phi$ and so cannot maximize $P(Y =y \mid X=x)$ directly, instead we write it as $P(Y=y) \,P(X=x \mid Y=y)$ using Bayes (ignoring the constant denominator) and we will choose some distribution to approximate these terms instead.
* Our classifier will lose the status of the optimal classifier once we do this.
* We can think of $P(Y=y)$ as the class prior which could be just the base rate of how the classes are distributed e.g. if we have training data with 65% in class 1 and 35% in class 2 we can just set $P(Y=1) = 0.65$ and $P(Y=2) = 0.35$ and call it a day.
* $P(X=x \mid Y=y)$ is our data likelihood and we can think of it as a class conditional distribution of our $X$ data given which class we are in. This is big departure from what we have done before where we have now made an assumption about how our covariates are distributed.
  * Note: $P(X=x \mid Y=y)$ is assuming $X$ is discrete, if it's continuous we can just use the class conditional density $P(x \mid Y=y)$.
* This type of classifier is called a generative model as we are modelling both the distributions of $X$ and $Y$. The generative part refers to the fact we will now be able to **generate new labelled data(!)**.
* Note: now we have approximated $P(Y=y)$ and $P(X=x \mid Y=y)$ we can no longer say that the resulting classifier will be optimal.
* **Naive Bayes Classifier (NBC)**
  * We could choose to define our class conditional density $P(x \mid Y=y) = N(x \mid u_y, \Sigma_y)$. In other words as a Gaussian with a different mean and covariance for each class. The NBC makes the assumption that the covariates $X$ are conditionally independent given $y$. In other words, there is no correlation between the features given the class. This allows us to write the class conditional densities as a product of one dimensional densities.
  * Whilst we do not expect these assumptions to hold the NBC often performs well in practice, one reason being that the simple model has few parameters and so is reasonably robust to over-fitting.

##### A note on the difference between discriminative and generative modelling (not covered in lecture):
* Loosely speaking **generative** models focus on building a model for our $X$ data which we are then able to sample from. This is very different from what we have been doing so far by learning $P(Y \mid X)$ which is called **discriminative** modelling and puts all its effort into trying to learn these conditional distributions and the explicit boundaries between our data classes. We typically call discriminative any classifiers that aren't generative.
* As a consequence, generative models impose structural assumptions onto your data which discriminative models do not. Generative models often outperform discriminative models on small data because the assumptions place some structure on your model that can help prevent over-fitting.
* That said, if we have a lot of data we usually achieve lower error rates by modelling $P(Y \mid X)$ directly as in the discriminative case.
* **Q: Why do we not use a generative model for (linear) regression?**
    * Note: I'm not 100% on this answer as it's something I'm still thinking about.
  * There is nothing to stop us using a generative model though estimating the covariance matrix $\Sigma$ requires estimating $\dfrac{(d+1)(d+2)}{2}$ parameters, plus the $d+1$ parameters for the mean.
  * Stack exchange [question](https://stats.stackexchange.com/questions/12421/generative-vs-discriminative?rq=1) on the topic.
* Further reading: no less a hero that Andrew Ng has a paper on the topic of discriminative vs. generative classifiers [topic](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf).

##### Linear classifiers classify our data according to which side of the decision boundary they fall on (and assume the data is linearly separable)
In the binary classification sense where our data is linearly separable a linear classifier works by classifying points according to which side of the hyperplane/boundary they fall (we are assuming linear separability of the data).
* In the binary case where we assume $P(x \mid y) = N(x \mid u_y, \Sigma)$ then this leads to a linear decision boundary and is called **Linear Discriminant Analysis (LDA).**
  * Note that we are assuming our $x$ data given the class $y$ shares the same covariance but has its own class mean (so think of two multivariate Gaussians just centred at different locations).
  * In this case we classify points according to the sign of $x^Tw + w_0$.
* If we assume $P(x \mid y) = N(x \mid u_y, \Sigma_y)$ then we end up with a quadratic decision boundary and this is called **Quadratic Discriminant Analysis (LDA)** although it is still linear in the weights (think of polynomial classification/regression).
* It is important to note that in the case of LDA and QDA we have explicit formula for $w$ and $w_0$ analogous to the LS setting where we can solve things exactly.

<hr class="with-margin">
<h4 class="header" id="math">Main mathematical ideas from the lectures</h4>

* **Gaussian class conditional densities**
   * Defining $p(x \mid Y=y) = N(x \mid \mu_y, \Sigma_y)$ then we can calculate the MLE estimates of $(\mu_y, \Sigma_y)$ which are just the empirical mean and covariances of the corresponding class $y$.
* **Using log odds to make a binary classification decision**
  * For a given class we are approximating $p(Y=y \mid X=x)$ with $p(x \mid Y=y)\,p(Y=y)$. For two classes taking the natural log of the ratio of the probabilities is called the log odds:
    * E.g. Evaluate if: $\ln \dfrac{p(x \mid y=1)\,p(y=1)}{p(x \mid y=0)\,p(y=0)} > 0$
    * **Example for LDA:** i.e. $p(x \mid Y=y) = N(x \mid \mu_y, \Sigma)$ this evaluates to something a little ugly but the main point to note is that there is a term not involving $x$ which we call $w_0$ and a term involving $x^T$ multiplied by a vector not involving $x$, we call this $w$. Both $w$ and $w_0$ involve $\Sigma, \mu_1, \mu_0$ as well as $\pi_0, \pi_1$ which are the baseline priors e.g. $p(y=0), p(y=1)$.
    * So evaluating which class a point belongs to is equivalent to determining the sign of $x^Tx + w_0$ where in this case we have explicit formula for $w$ and $w_0$.
    * This produces a linear decision boundary.
    * Extend to QDA (still solvable analytically) by using different covariances for each class.
* **Hyperplanes**
  * The main idea is that $x^Tw + w_0$ gives a sense of distance from the hyperplane with the sign telling us which side we are on. The lecture notes have some nice illustrations of this that are much easier to follow than words so please refer to those.

<hr class="with-margin">
<h4 class="header" id="math">Some mathematical details</h4>

Here are some of the mathematical details from the week:

* **Notions of distance**:
  * The typical Euclidean distance in $\mathbb{R}^d$ is $\|\|u-v\|\|\_2$ but we can extend this notion to the $l_p$ distance for $p \in [1, \infty]$ as follows: $l_p = \|\|u-v\|\|\_p = \big(\sum_{i=1}^{d}\|u_i - v_i\|^p\big)^{\frac{1}{p}}$
    * Here the single $\|$ means take the absolute value of the resulting number when we subtract $v_i$ from $u_i$.
* **Probability details**
  * There were two of these introduced in the lecture and whilst I don't usually find some aspects of probability theory highly intuitive I do think the latter of the two results mentioned in class actually has a simpler explanation. It was presented as follows:
    * $ C = E[A \mid B] $ with $A$ and $B$ both random, so $C$ is random.
    * $E[C] = E[E[A \mid B]] = E[A]$ is the 'tower property' of expectation
    * **What does this mean intuitively?**
    * **Example**: let's say there is a factory that makes fancy new GPUs and depending on the result of a fair coin flip some burn out after an average of 10,000 hours (flip H) and some burn out after an average of 20,000 hours (flip T).
      * Let's call flipping H event $B$ and random variable $A$ how many hours before the GPU burns out.
        * Note: $B^c$ is thus the event of flipping T.
      * In this case $ C = E[A \mid B] $ is still a random variable as given we have a GPU there is still some randomness as $A$ is itself random. However we can think about what the **average** value of $C$ might be, that is, the average numbers of hours a GPU from this factory will last: $E[C] = E[A \mid B]P[B] + E[A \mid B^c]P[B^c] = 10,000\cdot0.5 + 20,000\cdot0.5 = 15,000$ hours. This is no longer random.
      * It is just the probability weighted average and is a completely natural thing to do!
* **Cosine similarity and hyperplanes**
  * Note: this was incorrectly stated as the cosine rule in lecture which means something else. Cosine similarity is a measure of how 'similar' two vectors are my calculating the angle between them - vectors that point in the similar directions have a higher cosine similarity.

<hr class="with-margin">
<h4 class="header" id="sec3">Things I'm unclear on (or outstanding questions)</h4>

TBC

<hr class="with-margin">
<h4 class="header" id="sec4">What did the textbooks say?</h4>


To be updated.
