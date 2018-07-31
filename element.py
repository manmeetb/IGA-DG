"""
Module: element.py
------------------------------------------

Holds the element class. This will coorespond to the element on 
the reference domain.

NOTE: For now, only 1D cases can be handled
"""

import numpy
import basis
import integrate
import matplotlib.pyplot as plt


class ReferenceElement(object):

	"""
	Class for holding all there reference element information. This will hold
	data that is essentially common for every element that has this reference element
	"""

	def __init__(self, basis_ref_domain):

		"""
		Constructor for the ReferenceElement class

		:param basis_ref_domain: The Basis class object. Should have member variables
			for the basis_functions and derivative_basis_functions. Should 
			also have the basis defined on the reference domain.
		"""

		self.basis_ref_domain = basis_ref_domain  # The basis functions on the reference domain
		self.n = self.basis_ref_domain.n  # Number of basis functions/modal coefficients

		# The mass matrix where the L2 inner product is over the reference domain
		self.mass_matrix_ref_domain = self.get_mass_matrix() 

		# The modified mass matrix with elements of the form {phi_i * d_phi_j}
		self.mass_derivative_matrix_ref_domain = self.get_mass_derivative_matrix()


	def get_mass_matrix(self):

		"""
		Generate the mass matrix. If there are N_Basis basis functions, phi_i, then
		the mass matrix is given by M_ij = [ (phi_i,phi_j)_L2 ], where (:, :)_L2 is an 
		L2 inner product over the reference domain.

		:return: The mass matrix (as a numpy matrix)
		"""

		n = self.n  # number of basis functions
		M = numpy.zeros((n, n))

		# The space of functions (Vh). Is a list containing the lambda expressions for the 
		# basis functions
		function_space_Vh = self.basis_ref_domain.basis_functions
		quadrature_n = self.basis_ref_domain.p + 1  # Number of nodes/weights to use for the quadrature
		composite_quadrature_break_points = self.basis_ref_domain.composite_quadrature_break_points

		for i in range(n):
			for j in range(i+1):

				func = lambda x, i=i, j=j: function_space_Vh[i](x) * function_space_Vh[j](x)
				M[i][j] = integrate.integrate_gauss_legendre_quadrature(func, quadrature_n, -1.0, 1.0, 
					interval_break_points=composite_quadrature_break_points)

				M[j][i] = M[i][j]

		return M


	def get_mass_derivative_matrix(self):

		"""
		Generate the mass derivative matrix. If there are N_Basis basis functions, phi_i, then
		the mass derivative matrix is given by M_ij = [ (phi_i, dphi_j)_L2 ], where (:, :)_L2 is an 
		L2 inner product over the reference domain.

		:return: The mass derivative matrix (as a numpy matrix)
		"""

		n = self.n  # number of basis functions
		M = numpy.zeros((n, n))

		# The space of functions (Vh) and their derivatives (dVh). Is a list containing the 
		# lambda expressions for the basis functions
		function_space_Vh = self.basis_ref_domain.basis_functions
		function_space_dVh = self.basis_ref_domain.derivative_basis_functions

		quadrature_n = self.basis_ref_domain.p + 1  # Number of nodes/weights to use for the quadrature
		composite_quadrature_break_points = self.basis_ref_domain.composite_quadrature_break_points

		for i in range(n):
			for j in range(n):

				func = lambda x, i=i, j=j: function_space_Vh[i](x) * function_space_dVh[j](x)
				M[i][j] = integrate.integrate_gauss_legendre_quadrature(func, quadrature_n, -1.0, 1.0, 
					interval_break_points=composite_quadrature_break_points)

		return M


	def get_load_matrix(self, function_exact):

		"""
		Get the load matrix, b, used for performing an L2 projection. The function to project
		is function_exact. The b matrix contains one column and is of the form b_i = [ (f, phi_i)_L2 ], 
		where (:, :)_L2 is an L2 inner product over the reference domain.

		:param function_exact: The function to project using an L2 projection. The domain of interest
			is the reference domain ([-1, 1])

		:return: The load matrix b (will be a matrix with only one column, so essentially it is 
			a vector)
		"""
		
		n = self.n  # number of basis functions
		b = numpy.zeros((n, 1))

		# The space of functions (Vh) to perform the L2 projection onto
		function_space_Vh = self.basis_ref_domain.basis_functions

		quadrature_n = self.basis_ref_domain.p + 1  # Number of nodes/weights to use for the quadrature
		composite_quadrature_break_points = self.basis_ref_domain.composite_quadrature_break_points

		for i in range(n):
			func = lambda x, i=i: function_space_Vh[i](x) * function_exact(x)
			b[i][0] = integrate.integrate_gauss_legendre_quadrature(func, quadrature_n, -1.0, 1.0, 
					interval_break_points=composite_quadrature_break_points)

		return b


	def get_phi_ref_domain(self, phi_hat, xi_hat):

		"""
		Get the numerical solution evaluated at the point xi_hat on the
		reference domain.
	
		:param phi_hat: The coefficients of the basis functions
		:param xi_hat: The location on the reference domain to evaluate the solution
		"""

		val = 0.0

		for i in range(self.n):
			val += phi_hat[i] * self.basis_ref_domain.basis_functions[i](xi_hat)

		return val

