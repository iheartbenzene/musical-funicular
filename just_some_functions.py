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

def negative_binomial(k, r, p):
    return (gamma(k + r) / (factorial(k)*gamma(r))) * (p ** r) * ((1-p) ** k)

def probability_of_success():
    pass
