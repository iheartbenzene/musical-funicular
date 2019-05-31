import numpy as np
import scipy

def factorial(n):
    if n < 0 and n%2 == 0:
        pass
    elif n < 0 and n%2 != 0:
        pass
    elif n == 0:
        return 1
    return n * factorial(n-1)