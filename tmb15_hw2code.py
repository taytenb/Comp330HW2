import math
import numpy as np
from scipy.stats import invgamma
# load the data and put it in a dictionary
allData = {}
with open('data.txt', 'r') as data:
    for line in data:
        vals = [float(x) for x in line.split()]
        allData[int(vals[0])] = (vals[1], vals[2])
# parameters on the prior for m
mu_zero_m = 5.0
sigma_zero_m = 10.0
# parameters on the prior for c
mu_zero_c = 50.0
sigma_zero_c = 100.0
# parameters on the prior for sigma^2
a = 10.0
b = 1.0
# initial estimates for the three model parameters
m = 20.0
c = 50.0
sigma = 200.0


# write this for 1a)
def SampleSigma ():
    global allData, m, c, a, b
    n = len(allData)
    sum = 0.0
    for key, (height, weight) in allData.items():
        diff = weight - (height * m + c)
        sum += diff * diff
    aval = a + n/2.0
    bval = b + sum/2.0
    sigma = np.sqrt(invgamma.rvs(a=aval, scale=bval))
    return sigma
print("Sample Sigma outputs: \n")
for x in range(10):
    print(SampleSigma ())
    

# write this for 1b)
def SampleC ():
    global allData, m, sigma, mu_zero_c, sigma_zero_c
    n = len(allData)
    sum = 0.0
    for key, (height, weight) in allData.items():
        sum += weight - height * m
    sigma2 = sigma_zero_c * sigma_zero_c
    likelihood = sigma*sigma
    var = 1.0 / (1.0 / sigma2 + n / likelihood)
    mean = var * (mu_zero_c / sigma2 + sum / likelihood)
    c = np.random.normal(mean, np.sqrt(var))
    return c

print("Sample C outputs: \n")
for x in range(10):
    print(SampleC ())
    

# write this for 1c)
def SampleM ():
    global allData, c, sigma, mu_zero_m, sigma_zero_m
    n = len(allData)
    var = sigma_zero_m * sigma_zero_m
    sum = 0.0
    weightedsum = 0.0
    for key, (height, weight) in allData.items():
        sum += (height * height)/(sigma*sigma)
        weightedsum += (height * (weight - c))/(sigma*sigma)
    var2 = 1.0 / (1.0 / var + sum)
    mean = var2 * (mu_zero_m / var + weightedsum)
    m = np.random.normal(mean, np.sqrt(var2))
    return m
    
print("Sample M outputs: \n")
for x in range(10):
    print(SampleM ())
    
    
# this computes the error of the current model
def getError ():
    error = 0.0
    count = 0
    for x in allData:
        y = allData[x]
        error += (c + y[0] * m - y[1]) * (c + y[0] * m - y[1])
        count += 1
    return error / count
# for part 2, you run 1000 iteratins of a Gibbs sampler

errors = []
for x in range(1000):
    err = getError ()
    errors.append(err)
    sigma = SampleSigma ()
    m = SampleM ()
    c = SampleC ()
    
print("\nGibbs Sampler Results:")
print("First five errors:", errors[:5])
print("Last five errors:", errors[-5:])
print("Final parameter values:")
print("m =", m, "c =", c, "sigma =", sigma)