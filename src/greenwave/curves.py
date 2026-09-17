"""
Curve definitions.

Can interchange them when fitting the farm-seasons.
The m argument allows you to be package agnostic.
Can run the function under numpy ("np") or pytensor ("pt").
Pytensor is for PyMC which requires gradients to walk the parameter
space efficiently. Pytensor saves the curve expression as a computation graph
or calculation recipe which enables automatic differentiation without numerical 
approximation.
Numpy compute the curve's output immediately from the inputs and set parameters.
"""
import numpy as np
import pytensor.tensor as pt


def logistic(t, A, k, t0, m=np):
    """
    Symmetric sigmoid. A=plateau, k=growth rate, t0=inflection day.
    """
    # look back at this
    return A / (1.0 + m.exp(-k * (t - t0)))


def gompertz(t, A, k, t0, m=np):
    """
    Asymmetric sigmoid (long right taper, fast start, slow finish).
    Same parameter names as logistic.
    """
    return A * m.exp(-m.exp(-k * (t - t0)))

# Current curve options... can explore more or non-parametric curves too.
# Logistic and gompertz are for the yield(date) curves.
# growth_rate(date) could be fitted to a normal/gaussian curve.
CURVES = {"logistic": logistic, "gompertz": gompertz}
