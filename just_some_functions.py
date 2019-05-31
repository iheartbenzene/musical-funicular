import numpy as np
import scipy
from math import gamma

def factorial(n):
    if n < 0 and n%2 == 0:
        print("Please enter a non-negative integer.")
    elif n < 0 and n%2 != 0:
        print("Please enter a non-negative integer.")
    elif n == 0 or n == 1:
        return 1
    return n * factorial(n-1)

def negative_binomial(successes, failures, trials):
    '''
    The number of sucesses.
    The number of failures.
    The number of trials.
    '''
    return (gamma(trials) / (factorial(successes)*gamma(failures))) * (probability_of_success(successes, trials) ** failures) * ((1-probability_of_success(successes, trials)) ** successes)

def probability_of_success(successes, trials):
    '''
    The number of successes in the number of trials
    '''
    return successes / trials
