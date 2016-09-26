
*This is the notes I took going through dataquest Step 5 Statistics and Linear Algebra, it is also a practice for me to learn how to write Markdown and use Markdown Preview package in Sublime*

#Probability And Statistics In Python: Beginner
Skew
:	is a measure of the asymmetry of the probability distribution of a real-valud random variable about its mean. The skewness value can be positive or negative, or even undefined.

Kurtosis
:	measures whether the distribution is short and flat, or tall and skinny. 
	short -> platykurtic
	tall -> leptokurtic
	in between -> mesokurtic

Modality
:	refers to the number of modes, or peaks, in a distribution. Real-world data is often unimodal (only had one mode)

Mean
:	is a measure of central tendency, the `.mean()` method on Pandas series can be used to calculate variance.
```
# The axvline function will plot a vertical line over an existing plot
plt.axvline(value here)
```

Median
:	is a measure of central tendency, the `.median()` method on Pandas series can be used to calculate variance.

Variance
:	how spread out the data is around the mean, the `.var()` method on Pandas series can be used to calculate variance.
$$\sigma^2 = \frac{\displaystyle\sum_{i=1}^{n}(x_i - \mathbf{\bar{x}})^2} {n}$$

Standard Deviation
:	square root of the variance. `It is typical to measure what percentage of the data is within 1 standard deviation of the mean, or two standard deviations of the mean`. `std` metnod can be used on any Pandas `Dataframe` or `Series` to calculate the standard deviation
$$\sigma = \sqrt{\frac{\displaystyle\sum_{i=1}^{n}(x_i - \mathbf{\bar{x}})^2} {n}}$$

Normal distribution
:	
```python
import numpy as np
import matplotlib.pyplot as plt
# The norm module has a pdf function (pdf stands for probability density function)
from scipy.stats import norm

# The arange function generates a numpy vector
# The vector below will start at -1, and go up to, but not including 1
# It will proceed in "steps" of .01.  So the first element will be -1, the second -.99, the third -.98, all the way up to .99.
points = np.arange(-1, 1, 0.01)

# The norm.pdf function will take points vector and turn it into a probability vector
# Each element in the vector will correspond to the normal distribution (earlier elements and later element smaller, peak in the center)
# The distribution will be centered on 0, and will have a standard devation of .3
probabilities = norm.pdf(points, 0, .3)

# Plot the points values on the x axis and the corresponding probabilities on the y axis
# See the bell curve?
plt.plot(points, probabilities)
plt.show()
```

Measuring Correlation
:	r-value. A 1 means perfect positive correlation. A 0 means no correlation. a -1 means perfect negative correlation
```
from scipy.stats.stats import pearsonr

# The pearsonr function will find the correlation between two columns of data.
# It returns the r value and the p value.  We'll learn more about p values later on.
r, p_value = pearsonr(data1, data2)
# As we can see, t
his is a very high positive r value -- close to 1
print(r)
```

Calculate Covariance
:	`cov` metnod can be used from NumPy to compute covariance, returning a 2x2 matrix.
:	Covariance is how things vary together
$$cov(\mathbf{x},\mathbf{y})=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{n}$$


Calculate Correlation
:	
$$r = \frac{cov(\mathbf{x},\mathbf{y})}{\sigma_{x}\sigma_{y}}$$

```python
from numpy import cov
# The nba_stats variable has already been loaded.
covar= cov(nba_stats["fta"], nba_stats["blk"])[0,1]
r_fta_blk = covar/nba_stats["fta"].std()/nba_stats["blk"].std()
```

Linear regression and **standard error**
:		
$$RSS = \sum\limits_{i=1}^n{(y_{i} - \hat{y}_{i})^2}$$
$$standard error = \sqrt{\frac{RSS}{n - 2}}$$
```python
from scipy.stats import linregress
import numpy as np
slope, intercept, r_value, p_value, stderr_slope = linregress(x, y)
```

Distribution and sampling
:	
```python
import random

# Returns a random integer between the numbers 0 and 10, inclusive.
num = random.randint(0, 10)

# Generate a sequence of 10 random numbers between the values of 0 and 10.
random_sequence = [random.randint(0, 10) for _ in range(10)]

# Sometimes, when we generate a random sequence, we want it to be the same sequence whenever the program is run.
# An example is when you use random numbers to select a subset of the data, and you want other people
# looking at the same data to get the same subset.
# We can ensure this by setting a random seed.
# A random seed is an integer that is used to "seed" a random number generator.
# After a random seed is set, the numbers generated after will follow the same sequence.
random.seed(10)
print([random.randint(0,10) for _ in range(5)])
```
```python
# Let's say that we have some data on how much shoppers spend in a store.
shopping = [300, 200, 100, 600, 20]

# We want to sample the data, and only select 4 elements.

random.seed(1)
shopping_sample = random.sample(shopping, 4)
```



#Probability And Statistics In Python: Intermediate
##Calculating probabilities
$$Combination = \frac{N!}{k!(N-k)!}$$
$$Arrangement = \frac{N!}{(N-k)!}$$
$$Per Combination Probability = p^{k} * q^{N-k}$$

##Probability distribution
Binominal distribution
$$(p^{k} * q^{N-k}) * \frac{N!}{k!(N-k)!}$$
$$\mu = N * p$$
$$\sigma = \sqrt{N * p * q}$$

Probability mass function
	
	from scipy import linspace
	from scipy.stats import binom
	# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
	outcome_counts = linspace(0,30,31)
	# Create the binomial probabilities, one for each entry in outcome_counts.
	dist = binom.pmf(outcome_counts,30,0.39)

cumulative density function

	from scipy import linspace
	from scipy.stats import binom
	# Create a range of numbers from 0 to 30, with 31 elements (each number has one entry).
	outcome_counts = linspace(0,30,31)
	# Create the cumulative binomial probabilities, one for each entry in outcome_counts.
	dist = binom.cdf(outcome_counts,30,0.39)

z-scores
:	the number of standard deviations away from the mean a probability is
$$\frac{k - \mu}{\sigma}$$

##Significance testing
Null hypothesis
Alternative hypothesis

###Permutation test
:	The permutation test is a statistical test that involves simulating rerunning the study many times and recalculating the test statistic for each iteration. The goal is to calculate a distribution of the test statistics over these many iterations. This distribution is called the sampling distribution and it approximates the full range of possible test statistics under the null hypothesis. We can then benchmark the test statistic we observed in the data to determine how likely it is to observe this mean difference under the null hypothesis. If the null hypothesis is true, that the weight loss pill doesn't help people lose more weight, than the observed mean difference of 2.52 should be quite common in the sampling distribution. If it's instead extremely rare, then we accept the alternative hypothesis instead.
```python
empty = {"c": 1}
if empty.get("c", False):
    # If in the dictionary, grab the value, increment by 1, reassign.
    val = empty.get("c")
    inc = val + 1
    empty["c"] = inc
else:
    # If not in the dictionary, assign `1` as the value to that key.
    empty["c"] = 1
```



##P-value
:	P-value is widely used in statistical hypothesis testing, specifically in null hypothesis significance testing. The p-value is defined as the probability of obtaining a result equal to or "more extreme" than what was actually observed, when the null hypothesis is true. In frequentist inference, the p-value is widely used in statistical hypothesis testing, specifically in null hypothesis significance testing. In this method, as part of experimental design, before performing the experiment, one first chooses a model (the null hypothesis) and a threshold value for p, called the significance level of the test, traditionally 5% or 1% and denoted as α. **If the p-value is less than or equal to the chosen significance level (α), the test suggests that the observed data is inconsistent with the null hypothesis, so the null hypothesis must be rejected. However, that does not prove that the tested hypothesis is true. When the p-value is calculated correctly, this test guarantees that the Type I error rate is at most α. For typical analysis, using the standard α = 0.05 cutoff, the null hypothesis is rejected when p < .05 and not rejected when p > .05. The p-value does not in itself support reasoning about the probabilities of hypotheses but is only a tool for deciding whether to reject the null hypothesis.**
_A p-value allows us to determine whether the difference between two values is due to chance, or due to an underlying difference._

##Positive: corresponds to rejecting null hypothesis
Netagive: corresponds to failing to reject null hypothesis


| Table of error types 	     		  | T0 (valid/True)                   | T0 (invalid/False)                |  
| ------------------------------------| ----------------- ----------------| ----------------------------------|  
| Reject 		                      | Type I error (False Positive)     | Correct inference (True Positive) |  
| Fails to reject                     | Correct inference (True Negative) | Type II error (False Negative)    |  


##how to calculate P-value
 - decide on a test statistic, which is a numerical value that summarizes the data and we can use in statistical formulas.
 - generate a sampling distribution

how to calculate chi-squared (significance level)


numpy.random.random()      Pass (32561,) into the numpy.random.random function to get a vector with 32561 elements.why?
numpy.random.rand()


Vector is a one dimensional array

A tuple is a sequence of immutable Python objects. Tuples are sequences, just like lists. The differences between tuples and lists are, the tuples cannot be changed unlike lists and tuples use parentheses, whereas lists use square brackets.






#Lineary Algebra In Python
_Premium account needed_