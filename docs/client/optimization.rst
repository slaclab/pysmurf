A.. _tuning optimization:

tuning optimization
===================

serial_gradient_descent
-----------------------
.. automodule:: pysmurf.client.command.smurf_command
		:members:

- Gradient Descent

  This finds the minimum of the resonator using the stochastic gradient descent method/algorithm.

  Extra stuff/notes:

  - This is an optimization function
  - you can use it to approach the minimum of any differentiable function
  - it can get stuck in a local minimum instead of finding the global minimum but it is still useful
  - the cost/loss function is the function to be minimized (in this case it would be the function described by the resonator dip?)
  - goal is to minimize the difference between the predicted function and the actual data
  - minimize the sum of squared residuals (SSR)
  - working with gradient descent just means you're looking for the fastest decrease in the cost/loss function which is determined by the negative gradient
  - 
