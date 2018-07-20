
"""
Module: Basis.py
------------------------------------------

Handles generating the set of B Spline basis functions.
"""


import numpy
import matplotlib.pyplot as plt
import math


# Used for floating point comparisons
CONST_EPS = 1E-15
CONST_NUM_PLOT_PTS = 100
CONST_MAX_SMOOTHING_ITERATIONS = 40

class BSplineBasis(object):

	"""
	Class for holding the B Spline basis functions as lambda expressions.
	"""

	def __init__(self, p, n, knot_vector=None):

		"""
		Class Constructor

		:param p: The order of the basis functions
		:param n: The number of basis functions
		:param knot_vector: (Optional) The knot vector to use for the basis functions
		"""

		# Spline space parameters
		self.p = p
		self.n = n

		self.m = self.n + self.p + 1
		self.k = self.n - self.p


		if knot_vector is None:
			# Create an equally spaced, open knot vector. The knots will
			# span from -1 to 1
			self.knot_vector = []

			for i in range(self.p+1):
				self.knot_vector.append(-1.)

			num_non_end_knots = self.m - 2*(self.p+1)
			delta_knot = 2./(num_non_end_knots+1)

			for i in range(1, num_non_end_knots+1):
				self.knot_vector.append(-1. + i*delta_knot)

			for i in range(self.p+1):
				self.knot_vector.append(1.)
		
		else:
			# Knot vector provided
			self.knot_vector = knot_vector


		# Create the lambda expressions for the basis functions and add it to the N_p list
		self.basis_functions = []
		for i in range(self.n):
			N_ip_xi = lambda xi, i=i: self.N_ip(i, self.p, xi, self.knot_vector)
			self.basis_functions.append(N_ip_xi)


	# =====================================
	#			  Static Methods
	# =====================================

 	@staticmethod
	def N_ip(i,p,t,tVector):

		"""
		The B Spline basis function. This function evaluates the 
		1D basis function at a given point based on the order and 
		knot vector. The evaluation is done using the recursive definition 
		of the B spline basis. 

		NOTE : 
			- For the B Spline basis, any 0/0 term is defined to be 0. 
				Since floating point values are used, compare the numerator
				and denominator to an epsilon parameter.

		:param i: The ith basis function (B Spline) to use for the evaluation
		:param p: The order of the basis functions
		:param t: The value at which to evaluate the basis function at (will be
			in the domain of the knot vector).
		:param tVector: The knot vector (one dimension) for the basis.

		:return : Float for the value of the given basis function at the given point.
		"""

		# Handle the case where t is close to the edges
		if (abs(t-tVector[0]) < CONST_EPS):
			t = tVector[0] + CONST_EPS
		elif (abs(t-tVector[-1]) < CONST_EPS):
			t = tVector[-1] - CONST_EPS

		if p == 0:
			# Base case:

			if t < tVector[i+1] and t >= tVector[i]:
				return 1.
			else:
				return 0.
		
		else:
			# Recursive case:
			
			t_i = tVector[i]
			t_iPlus1 = tVector[i+1]
			t_iPlusP = tVector[i+p]
			t_iPlusPPlus1 = tVector[i+p+1]
				
			#	The first term
			num1 = (t - t_i) * BSplineBasis.N_ip(i, p-1, t, tVector)
			denom1 = t_iPlusP - t_i

			if abs(num1) < CONST_EPS and abs(denom1) < CONST_EPS:
				term1 = 0.
			else:
				term1 = num1/denom1

			#	The second term
			num2 = (t_iPlusPPlus1 - t) * BSplineBasis.N_ip(i+1, p-1, t, tVector)
			denom2 = t_iPlusPPlus1 - t_iPlus1
			if abs(num2) < CONST_EPS and abs(denom2) < CONST_EPS:
				term2 = 0.
			else:
				term2 = num2/denom2

			return term1 + term2


	# =====================================
	#				Getters
	# =====================================

	def get_smoothed_knots(self):

		# The equispaced "greville abscissae"
		x_bar = numpy.linspace(-1., 1., self.n)

		# Original knot vector
		xi = self.knot_vector[:]

		# Smoothed knot vectors
		xi_k = xi[:]
		xi_kPlus1 = xi[:]

		# Performing the smoothing iterations
		num_iters = 0
		while True:

			# Loop over the inner knots since
			for t in range(self.p+1, self.m-self.p-1):

				xi_kPlus1[t] = 0
				for i in range(self.n):
					xi_kPlus1[t] += x_bar[i]*self.N_ip(i, self.p, xi[t], xi_k)

			# Compute the L2 norm of the difference
			L2_Norm = 0
			for t in range(self.m):
				L2_Norm += (xi_kPlus1[t] - xi_k[t])**2
			L2_Norm = math.sqrt(L2_Norm)

			xi_k = xi_kPlus1[:]
			num_iters+=1

			if num_iters > CONST_MAX_SMOOTHING_ITERATIONS:
				raise ValueError("Max number of smoothing iterations reached")

			if L2_Norm < 1E-8:
				break

		return xi_kPlus1


	# =====================================
	#			Instance Methods
	# =====================================

	def plot_basis(self, color=None):

		"""
		Plot the set of basis functions. Plot the basis onto the global matplotlib plt
		object.

		:param color: (Optional) The matplotlib string for which color to make the curves
		:return: -
		"""

		xi_min = self.knot_vector[0]
		xi_max = self.knot_vector[-1]

		x_vals = numpy.linspace(xi_min, xi_max, CONST_NUM_PLOT_PTS)

		for N_b in self.basis_functions:

			y_vals = []

			for x_val in x_vals:
				y_vals.append(N_b(x_val))

			if color is None:
				plt.plot(x_vals, y_vals)
			else:
				plt.plot(x_vals, y_vals, c=color)


	def test_greville_abscissae(self):

		"""
		Method to test the greville abscissae and their properties with the b spline
		"""

		tau = []
		for i in range(self.n):
			tau_i = 0

			for j in range(self.p):
				tau_i += self.knot_vector[i+j+1]
			
			tau_i /= self.p

			tau.append(tau_i)
	
		print tau

		x_vals = numpy.linspace(-1, 1, 10)
		y_vals = []

		for x_val in x_vals:

			y_val = 0
			for i in range(self.n):
				y_val += tau[i]*self.basis_functions[i](x_val)

			y_vals.append(y_val)


def test_basis():

	b_spline_basis = BSplineBasis(3, 11)
	knots_smooth = b_spline_basis.get_smoothed_knots()

	b_spline_basis_smooth = BSplineBasis(3, 11, knots_smooth)


	plt.figure(1)
	b_spline_basis.plot_basis('r')
	print b_spline_basis.knot_vector
	b_spline_basis_smooth.plot_basis('b')
	print b_spline_basis_smooth.knot_vector
	plt.grid()
	plt.show(block=True)

	#b_spline_basis.test_greville_abscissae()


if __name__ == "__main__":
	test_basis()
