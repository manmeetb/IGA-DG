"""
Module: integrate.py
------------------------------------------

Module used to perform the gauss-legendre quadrature to integrate
given functions.
"""

import numpy


def integrate_gauss_legendre_quadrature(f, n, a, b, interval_break_points=None, nodes=None, weights=None):

	"""
	Integrate a given functional (with 1 input variable) over a given 
	domain [a,b]. Optionally, perform a composite gaussian quadrature over
	a set of intervals

	:param f: The function to integrate
	:param n: The number of nodes and weights to use for the integration. If a composite
		integration is being performed, this is the number of nodes and weights used on 
		each interval
	:param a: The lower limit of the integral
	:param b: The upper limit of the integral
	:param interval_break_points: (Optional) If a composite gaussian quadrature is required,
		this list holds the set of points, on the interval [a,b] to split the domain into.
		The list should include a and b, and it should be sorted in increasing order.
	:param nodes: (Optional) To speed up the computation and not have to compute the nodes and weights
		on each call to the function, take the nodes used for the quadrature (on the domain [-1,1]).
	:param weights: (Optional) To speed up the computation and not have to compute the nodes and weights
		on each call to the function, take the weights used for the quadrature.

	:return: The integral of the function f from a to b
	"""

	# The nodes and weights for the quadrature
	if nodes is None or weights is None:
		nodes, weights = numpy.polynomial.legendre.leggauss(n)

	if interval_break_points is not None:
		
		# Consider each subinterval and perform the quadrature over it recursively

		integral_value = 0

		for i in range(len(interval_break_points)-1):
			
			x_i = interval_break_points[i]
			x_iPlus1 = interval_break_points[i+1]

			integral_value += integrate_gauss_legendre_quadrature(f, n, x_i, x_iPlus1, nodes=nodes, weights=weights)
	else:

		# No subintervals to consider for the quadrature
		integral_value = 0

		# Transform integral onto [-1, 1] domain and perform the quadrature
		for i in range(n):
			integral_value += weights[i] * f(0.5*(b-a)*nodes[i] + 0.5*(b+a))
		integral_value *= 0.5*(b-a) 

	return integral_value


def integrate_midpoint_rule(f, a, b, n):

	# Used for testing
	# Integrate using the midpoint rule

	h = float(b-a)/n

	integral_value = 0.0
	for i in range(n):
		x_i = a + (i)*(h) + h/2.0
		integral_value += f(x_i)*h

	return integral_value


def main():

	f = lambda x: x**2
	print integrate_midpoint_rule(f, 0, 3, 200)


if __name__ == "__main__":
	main()




