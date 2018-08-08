
"""
Module: patch.py
------------------------------------------

Holds the patch class. This class instantiates patch objects to be used
for the multi-patch IGA code. Each patch will hold a set of elements which
are distinguished based on the knot vector.

NOTE: For now, only 1D cases can be handled
"""

import math
import element
import numpy
import parameters
import integrate


class Patch(object):

	"""
	Class for holding the Patch parameters and elements on which 
	to solve the PDE.

	NOTE: For now, only 1D patches will be able to be handled. 
	"""

	def __init__(self, x_vertices, reference_element, index):

		"""
		Class Constructor

		:param x_vertices: The limits of the patch on the physical domain.
			For these 1D cases, we have only one spatial coordinate (x)
		:param reference_element: The reference element for each patch on the mesh
		"""

		self.x_vertices = x_vertices  # Index 0 is left limit, index 1 is right limit
		self.reference_element = reference_element
		self.index = index
		self.J = 0.5 * (self.x_vertices[1] - self.x_vertices[0])  # Volume jacobian term


		# Mapping from the reference domain to the physical
		self.mapping_xi_to_x = lambda xi: (0.5*(self.x_vertices[1] - self.x_vertices[0])*xi + 
				0.5*(self.x_vertices[1] + self.x_vertices[0]))


		# The coefficients (and derivative) for the basis functions forming the numerical solution
		self.n = self.reference_element.n  # The number of modal coefficients
		self.phi_hat = [None]*self.reference_element.n
		self.phi_hat_p = [None]*self.reference_element.n  # Used for the rk4 time stepping
		self.dphi_hat = [None]*self.reference_element.n


		# Face Data structures
		self.patch_faces_ptr = [None]*2


		# RHS Data structures
		self.rhs = numpy.zeros((self.n,1))  # The RHS
		self.rhs_face = numpy.zeros((self.n,1))  # Face contribution for the RHS
		self.rhs_vol = numpy.zeros((self.n,1))  # Volume contribution for the RHS


		# LHS Data structures
		# - Store the inverse of the Mass matrix on the reference domain multiplied 
		# 	by the jacobian
		self.M_J_inv = self.reference_element.mass_matrix_ref_domain*self.J
		self.M_J_inv = numpy.linalg.inv(self.M_J_inv)


		# Global solve Data structures
		self.N_k, self.P_kMin1 = self.set_global_solve_matrices()


	def set_initial_condition(self, function_ic):

		"""
		Use an L2 projection to compute the coefficients, phi_hat, of the
		basis functions given a specified initial condition.

		:param function_ic: The initial condition function
		:param x_vertices: The limits on the 1D domain that this patch is
			defined
		"""

		M = self.reference_element.mass_matrix_ref_domain

		# Map the function onto the reference domain [-1,1]
		x1 = self.x_vertices[0]
		x2 = self.x_vertices[1]
		function_ic_ref = lambda xi: function_ic(0.5*(x2-x1)*xi + 0.5*(x2+x1))

		b = self.reference_element.get_load_matrix(function_ic_ref)

		coefficients = numpy.linalg.solve(M, b)

		for i in range(self.n):
			self.phi_hat[i] = coefficients[i][0]


	def get_phi(self, x):

		"""
		Get the approximate solution at the point x on the physical domain.

		:param x: The location on the physical domain to compute the solution at
		
		:return: The value of the numerical solution at the point x on the physical domain
		"""

		# Map the physical point (x) onto the reference domain
		xi_hat = 2.*(x - self.x_vertices[0])/(self.x_vertices[1] - self.x_vertices[0]) + -1
		return self.reference_element.get_phi_ref_domain(self.phi_hat, xi_hat)


	def set_RHS_to_zero(self):

		"""
		Set the RHS and its component matrices to zero
		"""

		for i in range(self.n):
			self.rhs[i][0] = 0.0
			self.rhs_face[i][0] = 0.0
			self.rhs_vol[i][0] = 0.0


	def add_RHS_face(self):

		"""
		Get the RHS face contribution.
		"""

		# Get the list of basis functions (defined on the reference domain)
		basis_functions_ref_domain = self.reference_element.basis_ref_domain.basis_functions

		# The numerical flux on the patch faces
		num_flux_l = self.patch_faces_ptr[0].numerical_flux
		num_flux_r = self.patch_faces_ptr[1].numerical_flux

		# Upwind Flux Approach

		for j in range(len(basis_functions_ref_domain)):
			
			rhs_face_j = 	num_flux_r * basis_functions_ref_domain[j](1.0) - \
							num_flux_l * basis_functions_ref_domain[j](-1.0)

			self.rhs_face[j][0] += (-1.0) * rhs_face_j # Subtract since moving contribution to the RHS

		"""
		# Set the a_sym values
		for j in range(len(basis_functions)):

			a_sym_j = 0.5 * ((num_flux_r*basis_functions[j](1.0)) - (num_flux_l*basis_functions[j](-1.0)))
			self.rhs[j][0] += (-1.0) * a_sym_j * (1/self.J) # Subtract since moving a_sym to RHS
		"""


	def add_RHS_volume(self):

		"""
		Get the RHS volume contribution
		"""

		beta = parameters.CONST_BETA  # Wave speed for linear advection

		# The mass derivative matrix on the reference domain. Therefore, is composed of the
		# L2 inner product over the reference domain and the basis functions and its 
		# derivative is mapped onto the reference domain 
		mass_derivative_matrix_ref_domain = self.reference_element.mass_derivative_matrix_ref_domain

		# Upwind Flux Approach
		for j in range(self.n):

			# Jacobian from the integral mapping cancels out with the jacobian that comes out 
			# through the gradient mapping to the reference domain

			rhs_vol_j = 0.0
			for i in range(self.n):
				rhs_vol_j += (-1.0) * beta * self.phi_hat[i] * mass_derivative_matrix_ref_domain[i][j]

			self.rhs_vol[j][0] += (-1.0) * rhs_vol_j  # Subtract since moving to the RHS

		"""
		for j in range(self.n):
			a_skew_j = 0.0
			for i in range(self.n):
				a_skew_j += -0.5*beta*self.phi_hat[i]*mass_derivative_matrix[i][j] + 0.5*beta*self.phi_hat[i]*mass_derivative_matrix[j][i]
			self.rhs[j][0] += (-1.0) * a_skew_j  # Subtract since moving a_skew to RHS
		"""


	def compute_RHS(self):

		"""
		Compute the RHS for the equation 

			dphi/dt = RHS

		Therefore, multiply by the inverse of the mass matrix to get the RHS vector
		to be used for the time stepping
		"""

		# Add the RHS Face and Volume contributions
		self.rhs = numpy.add(self.rhs_face, self.rhs_vol)

		# Multiply by the inverse mass matrix to compute the rhs
		self.rhs = numpy.dot(self.M_J_inv, self.rhs)


	def set_patch_face_pointer(self, patch_face, face_index):

		# face_index = 0 (left) or 1 (right)
		self.patch_faces_ptr[face_index] = patch_face


	def set_global_solve_matrices(self):

		"""
		Create the sub matrices that will be put into Lh for the 
		global solve.

		:return : N_k, P_kMin1
			N_k = Matrix that multiplies phi_hat on the RHS
			P_kMin1 = Matrix that multiplies phi_hat_min1 on the RHS (upwind flux)
		"""

		# Constant wave speed
		beta = parameters.CONST_BETA

		# Discretization Matrices
		M_k_inv = self.M_J_inv  # Inverse of mass matrix (with jacobian)
		S = None  # Stiffness matrix
		F_1 = numpy.zeros((self.n, self.n))  # First numerical flux matrix
		F_min1 = numpy.zeros((self.n, self.n))  # Second numerical flux matrix

		# Stiffness Matrix
		S = numpy.transpose(self.reference_element.mass_derivative_matrix_ref_domain)

		# Numerical flux matrices.
		# NOTE: Are symmetric so can optimize that here

		basis_functions_ref_domain = self.reference_element.basis_ref_domain.basis_functions 

		for i in range(self.n):
			for j in range(self.n):
				
				# beta * phi_i(1) * phi_j(1):
				beta_phi_i_phi_j_plus1 = beta * basis_functions_ref_domain[i](1.0) * basis_functions_ref_domain[j](1.0)   
				
				# beta * phi_i(-1) * phi_j(-1): 
				beta_phi_i_phi_j_min1 = beta * basis_functions_ref_domain[i](-1.0) * basis_functions_ref_domain[j](1.0)    

				F_1[i][j] = beta_phi_i_phi_j_plus1
				F_min1[i][j] = beta_phi_i_phi_j_min1

		N_k = beta * numpy.dot(M_k_inv, S) - numpy.dot(M_k_inv, F_1)
		P_kMin1 = numpy.dot(M_k_inv, F_min1)

		return N_k, P_kMin1


class PatchFace(object):

	"""
	Class for holding the Patch face information. In 1D, these is simply
	the point that distnguishes two patches.

	In a more general version, the patch face will be able to point to 
	BC faces. For now, everything will be periodic
	"""

	def __init__(self):

		"""
		PatchFace constructor
		"""

		# References to the left and right patch
		self.patch_ptr = [None]*2  # For now, have 2 (generalize for higher dimensional elements)

		# The numerical flux. For now, we only have one value, but this will
		# be generalized to higher dimensions eventually
		self.numerical_flux = None


	def set_patch_pointer(self, patch, face_index):

		"""
		Set the patch pointer based on the face index.
		"""

		self.patch_ptr[face_index] = patch

		if face_index == 0:
			# right face for patch, but relative to face the patch is to the left
			patch.set_patch_face_pointer(self, 1)
		else:
			# left face for patch, but relative to the face the patch is to the right
			patch.set_patch_face_pointer(self, 0)


	def compute_numerical_flux(self):

		"""
		Compute the numerical flux. For now, only the linear advection
		numerical flux will be supported.

		NOTE: Add upwinding flux first here
		"""

		"""
		# Energy Stable flux
		tau = parameters.TAU

		beta_left = parameters.BETA
		beta_right = parameters.BETA

		phi_left = self.patch_ptr[0].get_phi(self.patch_ptr[0].x_vertices[1])
		phi_right = self.patch_ptr[1].get_phi(self.patch_ptr[1].x_vertices[0])

		beta_avg = 0.5*(beta_left + beta_right)
		phi_jump = phi_right - phi_left

		self.numerical_flux = (beta_avg - tau * abs(beta_avg))*phi_jump
		"""

		# Upwind flux (wave always advects left to right)
		phi_left = self.patch_ptr[0].get_phi(self.patch_ptr[0].x_vertices[1])
			

		self.numerical_flux = parameters.CONST_BETA * phi_left




