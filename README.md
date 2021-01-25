# Bayesian Methods for Machine Learning (2020/2021)

Course given by [Simon Leglaive](https://sleglaive.github.io/) at [CentraleSupélec](https://www.centralesupelec.fr/en/).

## General information

Bayesian modeling, inference and prediction techniques have become commonplace in machine learning. Bayesian models are used in data analysis to describe, through latent factors, the generative process of complex data (medical images, audio, documents, etc.) The discovery of these latent or hidden variables from observations is based on the notion of posterior probability distribution, the calculation of which corresponds to the Bayesian inference step.

The Bayesian machine learning approach has the advantage of being interpretable, and it makes it easy to include expert knowledge through the definition of priors on the latent variables of interest. In addition, it naturally offers uncertainty information about the prediction, which can be particularly important in certain application contexts, such as medical diagnosis or autonomous driving for example.

At the end of the course, you are expected to: 

- know when it is useful or necessary to use a Bayesian machine learning approach; 
- have a view of the main approaches in Bayesian modeling and exact or approximate inference; 
- know how to identify and derive a Bayesian inference algorithm from the definition of a model; 
- be able to implement standard supervised or unsupervised Bayesian learning methods.

### Prerequisites

You are expected to be familiar with basic concepts of probabilities, statistics and machine learning. The 1st-year course "statistics and learning" provides all these requirements. 

We will have a session dedicated to the basics of statistical learning, so the most important is that you revise probabilities if you feel like you need to. To do so, you can read Chapter 6 "Probability and distribution" of [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf), by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, published by Cambridge University Press. 

### Bibliography

Most of the concepts that we will see in this course are discussed in the machine learning reference book [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), by Christopher M. Bishop, Springer, 2006, which is moreover freely available online.

Other useful references are:

- [Mathematics for Machine Learning](https://mml-book.github.io/book/mml-book.pdf), by Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, Cambridge University Press, 2020. (freely available online)
- Machine Learning: A Probabilistic Perspective, by Kevin P. Murphy, MIT Press, 2012. (available at the library).

### Evaluation

After each practical session, you should submit a report (in the form of a Jupyter notebook) with both theoretical and practical outcomes. The evaluation of these reports represents 30% of the final grade. The remaining 70% correspond to a final exam.

The exam is scheduled on Thursday, January 28, 9:30-11:30 (2 hours). All documents will be forbidden, but don't worry, if you need formulas such as the probability density functions of common distributions, it will be provided in an appendix.

### Anaconda install instructions

Not only for practicals but also during lectures, we will have to write/run Python scripts and more precisely Jupyter Notebooks. Before the first lecture, you have to install Anaconda (the individual edition) which is the easiest tool to manage Python packages. You can refer to Anaconda's documentation or follow a tutorial video (for instance, [this one in English](https://www.youtube.com/watch?v=5mDYijMfSzs&ab_channel=ProgrammingKnowledge) or [this one in French](https://www.youtube.com/watch?v=jaw5FhWx2Bk&ab_channel=MachineLearnia).).

With Anaconda, you can easily manage your Python packages. See for instance in the documentation how to install a new package, using Anaconda prompt (which is simply the terminal in Ubuntu or Mac OS).

If you have troubles installing Anaconda, which is very unlikely as it is super simple, you can run Jupyter Notebooks in Google Colab, which is a free online computing environment where you can run code on CPU but also GPU. Keep in mind however that working locally on your computer may be more ecological than using Google servers.

### Agenda

<table>
<tbody>
<tr>
<td>November 30</td>
<td>PM</td>
<td>Lecture</td>
<td>Fundamentals of Bayesian modeling and inference</td>
</tr>
<tr>
<td>December 4</td>
<td>AM</td>
<td>Lecture</td>
<td>Fundamentals of machine learning</td>
</tr>
<tr>
<td>December 9</td>
<td>AM</td>
<td>Lecture</td>
<td>Bayesian networks and inference in latent variable models</td>
</tr>
<tr>
<td>December 11</td>
<td>AM</td>
<td>Practical</td>
<td>Gaussian mixture model</td>
</tr>
<tr>
<td>December 18</td>
<td>AM</td>
<td>Lecture</td>
<td>Factor Analysis</td>
</tr>
<tr>
<td>January 6</td>
<td>AM</td>
<td>Lecture</td>
<td>Variational inference</td>
</tr>
<tr>
<td>January 8</td>
<td>AM</td>
<td>Practical</td>
<td>Bayesian linear regression</td>
</tr>
<tr>
<td>January 13</td>
<td>AM</td>
<td>Lecture</td>
<td>Markov Chain Monte Carlo</td>
</tr>
<tr>
<td>January 15</td>
<td>AM</td>
<td>Practical</td>
<td>Sparse Bayesian linear regression<br></td>
</tr>
<tr>
<td>January 18</td>
<td>AM</td>
<td>Lecture</td>
<td>Deep generative models</td>
</tr>
<tr>
<td>January 22</td>
<td>AM</td>
<td>Lecture</td>
<td>Revision and other activities</td>
</tr>
<tr>
<td>January 25</td>
<td>9:30-11:30</td>
<td>Exam</td>
<td><br></td>
</tr>
</tbody>
</table>

AM = 8:30-10:00 10:30-12:00

PM = 13:30-15:00 15:15-16:45

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
- [Notebook - Gaussian model with latent mean](session1/python/Gaussian%20model%20with%20latent%20mean.ipynb)

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

## Gaussian mixture model

This practical session is about the Gaussian mixture model, a generative model used to perform clustering, in an unsupervised fashion.

Material:

- [Notebook - Gaussian mixture model](session4/GMM.ipynb)
- [gmm_tools.py](session4/gmm_tools.py)
- [adventures_bayes.gif](session4/adventures_bayes.gif)
- [bayes_latents_1.png](session4/bayes_latents_1.png)


## Factor Analysis

Key concepts you should be familiar with at the end of this lecture:

- Factor analysis (obviously...)
- Factor analysis generative model 
- Derivation of the posterior
- Derivation of the marginal likelihood
- Properties of the multivariate Gaussian distribution
- Derivation of an EM algorithm (with continuous latent variables, contrary to the previous session on GMMs) for parameters estimation

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session5/slides.html)
- [Handwritten solution to the EM algorithm](session5/FA.pdf)

## Variational inference

Key concepts you should be familiar with at the end of this lecture:

- The problem of intractable posterior
- Kullback-Leibler divergence
- Variational inference 
- Mean-field approximation

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session6/slides.html)
- [Demo mean field variational inference](session6/demo-MF-VI.pdf)
- [Solution to exercices](session6/solution_exercises.pdf)

## Bayesian linear regression

In the second lecture, we already discussed about linear regression (polynomial regression) and we have seen that with a standard maximum likelihood approach, we have to carefully choose the degree of the polynomial model, in order not to overfit the training data. In Bayesian linear regression, a prior is consider for the weights, which acts as a regularizer and prevents overfitting. Moreover, this Bayesian approach to linear regression naturally provides a measure of uncertainty along with the prediction.

Material:

- [Notebook - Bayesian linear regression](session7/bayesian_LR.ipynb)
- [utils.py](session7/utils.py)


## Markov Chain Monte Carlo

Key concepts you should be familiar with at the end of this lecture:

- The Monte Carlo method to approximate expectations
- Sampling methods (inverse transform sampling, change of variable, rejection sampling, importance sampling)
- Definition of Markov chains (will not be at the exam)
- Markov chain Monte Carlo methods (will not be at the exam)

Material:

- [Slides](https://sleglaive.pages.centralesupelec.fr/page/bayesian-ML/session8/slides.html#1)

## Sparse Bayesian linear regression

This practical session is a follow-up of the previous one on Bayesial linear regression. We complexify the prior for the linear regression weights so that exact posterior inference is now intractable and a variational approach has to be developed.

Material:

- [Notebook - sparse Bayesian linear regression](session9/sparse_bayesian_LR.ipynb)
- [utils.py](session9/utils.py)
- [EM.py](session9/EM.py)

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

## Revision and other activities

**Activity 1**: Q&A session

**Activity 2**: You will find an exercise [here](session11/revisions.pdf). You will also find exercices (that were left as homeworks) in the slides of the different lectures.

**Activity 3**: If not finished yet, you can continue working on the sparse Bayesian linear regression notebook (the deadline to send your report is tonight).

**Activity 4**: Reading about sequential data processing with latent-variable models.

"State-space models (SSM) provide a general and flexible methodology for sequential data modelling. They were first introduced in the 1960s, with the seminal work of Kalman and were soon used in the Apollo Project to estimate the trajectory of the spaceships that were bringing men to the moon. Since then, they have become a standard tool for time series analysis in many areas well beyond aerospace engineering. In the machine learning community in particular, they are used as generative models for sequential data, for predictive modelling, state inference and representation learning". *Quote from Marco Fraccaro's Ph.D Thesis entitled "Deep Latent Variable Models for Sequential Data" and defended at Technical University of Denmark in 2018.*


The Kalman filter and smoother are used to compute the posterior distribution of a sequence of latent vectors (called the states) given an observed sequence of measurement. In the following video, a Kalman filter is used to track the latent position of multiple persons over time. The latent state variable in this case is continuous.
 
<iframe width="560" height="315" src="https://www.youtube.com/embed/bkn6M4LAoHk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
 
When the latent state variable is discrete, the state-space model is called a hidden Markov model (HMM). HMMs were very popular for automatic speech recognition, before the deep learning era. The latent state variable in this context is discrete and corresponds to a phoneme (an elementary unit of speech sound that allows us to distinguish one word from another in a particular language), while the observations are acoustic speech features computed from the audio signal.

Chapter 3 of Marco Fraccaro's Ph.D Thesis available [here](https://marcofraccaro.github.io/download/publications/fraccaro_phd_thesis.pdf) gives a very nice introduction to state-space models and Kalman filtering. To go a bit further, Chapter 4 introduces deep latent variable models for sequential data processing, using the framework of variational autoencoders.

## Acknowledgements

The slides are created using [Remark](https://github.com/gnab/remark/wiki), "A simple, in-browser, Markdown-driven slideshow tool". The template is modified from [Marc Lelarge](https://www.di.ens.fr/~lelarge/)'s template used in his (very very nice) [deep learning course](https://dataflowr.github.io/website/).

I did my best to clearly acknowledge the authors of the ressources that I could have been using to build this course. If you find any missing reference, please contact me. 

If you want to reuse some of the materials in this repository, please also indicate where you took it.

If you are not one of my student and you would like to have the solution to the practical works, you can contact me.

Email address: ```firstname.lastname@centralesupelec.fr```


## License

GNU Affero General Public License (version 3), see ```LICENSE.txt```.