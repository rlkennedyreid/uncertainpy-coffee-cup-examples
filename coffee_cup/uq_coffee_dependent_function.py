import chaospy
import numpy
import uncertainpy
from scipy.integrate import odeint


# Create the coffee cup model function
def coffee_cup_dependent(kappa_hat, T_env, alpha):
    # Initial temperature and time
    time = numpy.linspace(0, 200, 150)  # Minutes
    T_0 = 95  # Celsius

    # The equation describing the model
    def f(T, time, alpha, kappa_hat, T_env):
        return -alpha * kappa_hat * (T - T_env)

    # Solving the equation by integration.
    temperature = odeint(f, T_0, time, args=(alpha, kappa_hat, T_env))[:, 0]

    # Return time and model results
    return time, temperature


def model():
    # Create a model from the coffee_cup_dependent function and add labels
    model = uncertainpy.Model(
        coffee_cup_dependent, labels=["Time (s)", "Temperature (C)"]
    )

    # Create the distributions
    T_env_dist = chaospy.Uniform(15, 25)
    alpha_dist = chaospy.Uniform(0.5, 1.5)
    kappa_hat_dist = chaospy.Uniform(0.025, 0.075) / alpha_dist

    # Define the parameters dictionary
    parameters = {"alpha": alpha_dist, "kappa_hat": kappa_hat_dist, "T_env": T_env_dist}

    # We can use the parameters dictionary directly
    # when we set up the uncertainty quantification
    UQ = uncertainpy.UncertaintyQuantification(model=model, parameters=parameters)

    # Perform the uncertainty quantification,
    # which automatically use the Rosenblatt transformation
    # We set the seed to easier be able to reproduce the result
    data = UQ.quantify(seed=10)
