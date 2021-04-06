# Bayesian Methods for Machine Learning (2020/2021)

Course given by [Simon Leglaive](https://sleglaive.github.io/) at [CentraleSupélec](https://www.centralesupelec.fr/en/).

## General information

Bayesian modeling, inference and prediction techniques have become commonplace in machine learning. Bayesian models are used in data analysis to describe, through latent factors, the generative process of complex data (e.g. medical images, audio signals, text documents). The discovery of these latent or hidden variables from observations is based on the notion of posterior probability distribution, the calculation of which corresponds to the Bayesian inference step.

The Bayesian machine learning approach has the advantage of being interpretable, and it makes it easy to include expert knowledge through the definition of priors on the latent variables of interest. In addition, it naturally offers uncertainty information about the prediction, which can be particularly important in certain application contexts, such as medical diagnosis or autonomous driving for example.

At the end of the course, you are expected to: 

- know when it is useful or necessary to use a Bayesian machine learning approach; 
- have a view of the main approaches in Bayesian modeling and exact or approximate inference; 
- know how to identify and derive a Bayesian inference algorithm from the definition of a model; 
- be able to implement standard supervised or unsupervised Bayesian learning methods.

### Prerequisites

You are expected to be familiar with basic concepts of probabilities, statistics and machine learning. The 1st-year course "statistics and learning" at CentraleSupélec provides all these requirements. 

We will have a session dedicated to the basics of statistical learning, so the most important is that you revise probabilities if you feel like you need to. To do so, you can read Chapter 6 "Probability and distribution" of [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf), by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, published by Cambridge University Press. 

### Bibliography

Most of the concepts that we will see in this course are discussed in the machine learning reference book [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), by Christopher M. Bishop, Springer, 2006, which is moreover freely available online.

Other useful references are:

- [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf), by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, Cambridge University Press, 2020. (freely available online)
- Machine Learning: A Probabilistic Perspective, by Kevin P. Murphy, MIT Press, 2012. (available at the library).

### Agenda

<table>
<tbody>
<tr>
<td>November 30</td>
<td>PM</td>
<td>Lecture</td>
<td><a href='#session1'>Fundamentals of Bayesian modeling and inference</a></td>
</tr>
<tr>
<td>December 4</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session2'>Fundamentals of machine learning</a></td>
</tr>
<tr>
<td>December 9</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session3'>Bayesian networks and inference in latent variable models</a></td>
</tr>
<tr>
<td>December 11</td>
<td>AM</td>
<td>Practical</td>
<td><a href='#session4'>Gaussian mixture model</a></td>
</tr>
<tr>
<td>December 18</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session5'>Factor Analysis</a></td>
</tr>
<tr>
<td>January 6</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session6'>Variational inference</a></td>
</tr>
<tr>
<td>January 8</td>
<td>AM</td>
<td>Practical</td>
<td><a href='#session7'>Bayesian linear regression</a></td>
</tr>
<tr>
<td>January 13</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session8'>Markov Chain Monte Carlo</a></td>
</tr>
<tr>
<td>January 15</td>
<td>AM</td>
<td>Practical</td>
<td><a href='#session9'>Sparse Bayesian linear regression</a></td>
</tr>
<tr>
<td>January 18</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session10'>Deep generative models</a></td>
</tr>
<tr>
<td>January 22</td>
<td>AM</td>
<td>Lecture</td>
<td><a href='#session11'>Revision and other activities</a></td>
</tr>
<tr>
<td>January 28</td>
<td>9:30-11:30</td>
<td>Exam</td>
<td><br></td>
</tr>
</tbody>
</table>

AM = 8:30-10:00 10:30-12:00

PM = 13:30-15:00 15:15-16:45


<a id='session1'></a>
## Fundamentals of Bayesian modeling and inference

Key concepts you should be familiar with at the end of this lecture:

- Latent and observed variable
- Bayesian modeling and inference
- Prior, likelihood, marginal likelihood and posterior
- Decision, posterior expected loss
- Predictive prior, predictive posterior
- Gaussian model with latent mean or variance
- Conjugate, non-informative and hierarchical priors

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session1/slides.html)
- [Notebook - Gaussian model with latent mean](session1/python/Gaussian%20model%20with%20latent%20mean%20(solution).ipynb)
- [Handwritten solution to exercices](session1/solution_exercises.pdf)

<a id='session2'></a>
## Fundamentals of machine learning

Key concepts you should be familiar with at the end of this lecture:

- Supervised learning
- Empirical risk minimization
- Underfitting and overfitting
- Bias-variance trade-off
- Maximum likelihood, maximum a posteriori
- Multinomial logistic regression

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session2/slides.html)
- [Notebook - Multinomial logistic regression](session2/python/multinomial%20logistic%20regression.ipynb)

<a id='session3'></a>
## Bayesian networks and inference in latent variable models

Key concepts you should be familiar with at the end of this lecture:

- Bayesian network (or directed probabilistic graphical model)
- Conditional independence
- D-separation
- Markov blanket
- Generative model with latent variables
- Evidence lower-bound
- Expectation-maximization algorithm

Material:

- [Slides for part 1 - Bayesian networks](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session3a/slides.html)
- [Slides for part 2 - Inference in latent variable models](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session3b/slides.html)

<a id='session4'></a>
## Gaussian mixture model

This practical session is about the Gaussian mixture model, a generative model used to perform clustering, in an unsupervised fashion.

Material:

- [Notebook - Gaussian mixture model](session4/GMM.ipynb)
- [gmm_tools.py](session4/gmm_tools.py)

<a id='session5'></a>
## Factor Analysis

Key concepts you should be familiar with at the end of this lecture:

- Factor analysis generative model 
- Derivation of the posterior
- Derivation of the marginal likelihood
- Properties of the multivariate Gaussian distribution
- Derivation of an EM algorithm (with continuous latent variables, contrary to the previous session on GMMs) for parameters estimation

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session5/slides.html)
- [Handwritten solution to the EM algorithm](session5/FA.pdf)

<a id='session6'></a>
## Variational inference

Key concepts you should be familiar with at the end of this lecture:

- The problem of intractable posterior
- Kullback-Leibler divergence
- Variational inference 
- Mean-field approximation

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session6/slides.html)
- [Notebook - Variational inference for the Gaussian model with latent parameters](session6/python/VI%20for%20Gaussian%20model%20with%20latent%20mean%20and%20precision%20(solution).ipynb)
- [Handwritten solution to exercices](session6/solution_exercises.pdf)

<a id='session7'></a>
## Bayesian linear regression

We already discussed about linear regression (polynomial regression) in the second lecture, and we saw that with a standard maximum likelihood approach, we have to carefully choose the degree of the polynomial model in order not to overfit the training data. In Bayesian linear regression, a prior distribution is considered for the weights, which acts as a regularizer and prevents overfitting. Moreover, this Bayesian approach to linear regression naturally provides a measure of uncertainty along with the prediction.

Material:

- [Notebook - Bayesian linear regression](session7/bayesian_LR.ipynb)
- [utils.py](session7/utils.py)

<a id='session8'></a>
## Markov Chain Monte Carlo

Key concepts you should be familiar with at the end of this lecture:

- The Monte Carlo method to approximate expectations
- Sampling methods (inverse transform sampling, change of variable, rejection sampling, importance sampling)
- Definition of Markov chains
- Markov chain Monte Carlo methods

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session8/slides.html#1)

<a id='session9'></a>
## Sparse Bayesian linear regression

This practical session is a follow-up of the previous one on Bayesian linear regression. We complexify the prior for the linear regression weights so that exact posterior inference is now intractable and a variational approach has to be developed.

Material:

- [Notebook - sparse Bayesian linear regression](session9/sparse_bayesian_LR.ipynb)
- [utils.py](session9/utils.py)
- [EM.py](session9/EM.py)

<a id='session10'></a>
## Deep generative models

Key concepts you should be familiar with at the end of this lecture:

- The problem of (deep) generative modeling
- Generative model of the variational autoencoder (VAE), a non-linear generalization of factor analysis
- VAE inference model
- VAE training procedure
- Application of VAEs for MNIST image generation

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session10/slides.html)
- [Notebook - Variational autoencoder in PyTorch](session10/python/VAE.ipynb)
- [utils.py](session10/python/utils.py)

<a id='session11'></a>
## Revision and other activities

**Activity 1**: Q&A session

**Activity 2**: You will find an exercise [here](session11/revisions.pdf). You will also find exercices (that were left as homeworks) in the slides of the different lectures.

**Activity 3**: If not finished yet, you can continue working on the sparse Bayesian linear regression notebook (the deadline to send your report is tonight).

**Activity 4**: Reading about sequential data processing with latent-variable models.

"State-space models (SSM) provide a general and flexible methodology for sequential data modelling. They were first introduced in the 1960s, with the seminal work of Kalman and were soon used in the Apollo Project to estimate the trajectory of the spaceships that were bringing men to the moon. Since then, they have become a standard tool for time series analysis in many areas well beyond aerospace engineering. In the machine learning community in particular, they are used as generative models for sequential data, for predictive modelling, state inference and representation learning". *Quote from Marco Fraccaro's Ph.D Thesis entitled "Deep Latent Variable Models for Sequential Data" and defended at Technical University of Denmark in 2018.*


The Kalman filter and smoother are used to compute the posterior distribution of a sequence of latent vectors (called the states) given an observed sequence of measurement. In [this video](https://www.youtube.com/watch?v=bkn6M4LAoHk&feature=emb_title&ab_channel=MeteY%C4%B1ld%C4%B1r%C4%B1m), a Kalman filter is used to track the latent position of multiple persons over time. The latent state variable in this case is continuous.
 
When the latent state variable is discrete, the state-space model is called a hidden Markov model (HMM). HMMs were very popular for automatic speech recognition, before the deep learning era. The latent state variable in this context is discrete and corresponds to a phoneme (an elementary unit of speech sound that allows us to distinguish one word from another in a particular language), while the observations are acoustic speech features computed from the audio signal.

Chapter 3 of Marco Fraccaro's Ph.D Thesis available [here](https://marcofraccaro.github.io/download/publications/fraccaro_phd_thesis.pdf) gives a very nice introduction to state-space models and Kalman filtering. To go a bit further, Chapter 4 introduces deep latent variable models for sequential data processing, using the framework of variational autoencoders.

## Acknowledgements

The slides are created using [Remark](https://github.com/gnab/remark/wiki), "A simple, in-browser, Markdown-driven slideshow tool". The template is modified from [Marc Lelarge](https://www.di.ens.fr/~lelarge/)'s template used in his (very nice) [deep learning course](https://dataflowr.github.io/website/).

I did my best to clearly acknowledge the authors of the ressources that I could have been using to build this course. If you find any missing reference, please contact me. 

--- 

If you want to reuse some of the materials in this repository, please also indicate where you took it.

If you are not one of my student and you would like to have the solution to the practical works, you can contact me.

Email address: ```firstname.lastname@centralesupelec.fr```


## License

GNU Affero General Public License (version 3), see ```LICENSE.txt```.