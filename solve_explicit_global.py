
"""
Module: solve_explicit_global.py
------------------------------------------

Solve the PDE using an explicit solver. Do this by assembling a 
global discretization matrix (Lh) and then step all the solutions
together. Therefore, we are creating a global system, using a method
of line discretization, to create

	d/dt {phi} = [Lh] {phi} 

This global solve, for now, will only be valid for the 1D linear advection equation.
"""

import patch
import parameters
import math
import numpy
import sys
import solve_explicit


def solve_explicit_global(patches, patch_faces):

	"""
	Solve the equation using an explicit scheme and globally

	:param patches: The list of patches that make up the mesh
	"""

	time = 0.0

	for t_step in range(parameters.CONST_NUM_TIME_STEPS):

		# Perform the time step
		perform_global_time_step(patches, patch_faces)

		# Output Progress
		time += parameters.CONST_DELTA_T
		print "t_step: %d / %d, Time: %f " % (t_step+1, parameters.CONST_NUM_TIME_STEPS, time)
	

def perform_global_time_step(patches, patch_faces):

	"""
	Loop through the patches and perform the time step to obtain the 
	updated coefficient values for the next time step
	"""

	if parameters.CONST_TIME_STEP_METHOD == "EULER":
		euler_global_time_step(patches, patch_faces)
	elif parameters.CONST_TIME_STEP_METHOD == "EXPLICIT_RK4_LOW_STORAGE":
		explicit_rk4_lowstorage_global_time_step(patches, patch_faces)


def euler_global_time_step(patches, patch_faces):

	"""
	Perform an Euler time step to update the solution. 
	Do this by solving the problem globally
	"""

	# Assemble the global matrices
	Lh = assemble_Lh(patches)
	phi_hat_global_n = assemble_phi_hat(patches)

	# Perform the euler time step
	delta_t = parameters.CONST_DELTA_T
	dphi_hat_global_n = numpy.dot(Lh, phi_hat_global_n)
	phi_hat_global_nPlus1 = phi_hat_global_n + delta_t * dphi_hat_global_n

	# Update the modal coefficients
	update_phi_hat_patches(patches, phi_hat_global_nPlus1)

	# Check that solution is still bounded
	if not check_solution_bounded(patches):
		raise ValueError("NAN Solution Value")


def explicit_rk4_lowstorage_global_time_step(patches, patch_faces):

	"""
	Perform an explicit, low storage, fourth order Runge Kutta time 
	step to update the solution in time
	"""

	# Coefficients for the method
	rk4a = [0.0,
			-567301805773.0/1357537059087.0, 
			-2404267990393.0/2016746695238.0,
			-3550918686646.0/2091501179385.0, 
			-1275806237668.0/842570457699.0]

	rk4b = [1432997174477.0/9575080441755.0,
			5161836677717.0/13612068292357.0, 
			1720146321549.0/2090206949498.0,
			3134564353537.0/4481467310338.0,  
			2277821191437.0/14882151754819.0]

	rk4c = [0.0,
			1432997174477.0/9575080441755.0,
			2526269341429.0/6820363962896.0,
			2006345519317.0/3224310063776.0,
			2802321613138.0/2924317926251.0]

	delta_t = parameters.CONST_DELTA_T

	# Set p0: p0 = u_n
	for patch in patches:
		patch.phi_hat_p = patch.phi_hat[:]

	for rk in range(5):

		# Assemmble the RHS of the ODE system
		Lh = assemble_Lh(patches)
		phi_hat_global_n = assemble_phi_hat(patches)
		dphi_hat_global_n = numpy.dot(Lh, phi_hat_global_n)  # Global RHS

		for patch in patches:
			for i in range(patch.n):
				patch.phi_hat_p[i] *= rk4a[rk]
				patch.phi_hat_p[i] += delta_t * dphi_hat_global_n[patches.index(patch)*patch.n + i][0]
				patch.phi_hat[i]   += rk4b[rk] * patch.phi_hat_p[i]

	# Check that solution is still bounded
	if not check_solution_bounded(patches):
		raise ValueError("NAN Solution Value")


def assemble_Lh(patches):

	"""
	Create the discretization matrix (Lh) for the global 
	solve. 

	:return: The matrix Lh
	"""

	# Size of Lh:
	size_Lh = 0
	for patch in patches:
		size_Lh += patch.n

	# The empty Lh matrix to be filled
	Lh = numpy.zeros((size_Lh, size_Lh))

	# Add in the entries into the Lh matrix. 
	# NOTE: Assumes all elements have same number of dofs

	for patch in patches:

		patch_index = patches.index(patch)
		n = patch.n  # size of each square matrix (n = number of modal coefficients)

		A_k, B_k, C_k = patch.A_k, patch.B_k, patch.C_k

		# Get the ranges for slicing into Lh and setting the entries
		
		if patch_index == 0:
			# Periodic BCs
			A_i_range, A_j_range = [patch_index*n, (patch_index+1)*n], [(len(patches)-1)*n, (len(patches))*n]
		else:
			A_i_range, A_j_range = [patch_index*n, (patch_index+1)*n], [(patch_index-1)*n, (patch_index)*n]
		
		B_i_range, B_j_range = [patch_index*n, (patch_index+1)*n], [patch_index*n, (patch_index+1)*n]

		if patch_index == len(patches)-1:
			# Periodic BCs
			C_i_range, C_j_range = [patch_index*n, (patch_index+1)*n], [(0)*n, (1)*n]
		else:
			C_i_range, C_j_range = [patch_index*n, (patch_index+1)*n], [(patch_index+1)*n, (patch_index+2)*n]
	

		# Fill the array with the data for this patch
		Lh[A_i_range[0]:A_i_range[1], A_j_range[0]:A_j_range[1]] = A_k
		Lh[B_i_range[0]:B_i_range[1], B_j_range[0]:B_j_range[1]] = B_k
		Lh[C_i_range[0]:C_i_range[1], C_j_range[0]:C_j_range[1]] = C_k


	return Lh


def assemble_phi_hat(patches):

	"""
	Assemble the global phi_hat vector. 
	"""

	# Size of phi_hat:
	size_phi_hat = 0
	for patch in patches:
		size_phi_hat += patch.n

	phi_hat = numpy.zeros((size_phi_hat, 1))


	# Add the phi_hat entries for each patch into the 
	# global vector

	for patch in patches:

		patch_index = patches.index(patch)
		n = patch.n
		phi_hat_k = patch.phi_hat

		for i in range(n):

			global_i = patch_index*n + i
			phi_hat[global_i][0] = phi_hat_k[i]

		patch_index += 1

	return phi_hat


def update_phi_hat_patches(patches, phi_hat_global):

	"""
	Transfer the data from the global phi_hat vector in the phi_hat 
	lists for each patch
	"""

	for patch in patches:
			
		patch_index = patches.index(patch)
		n = patch.n
		phi_hat_k = patch.phi_hat

		for i in range(n):
			global_i = patch_index*n + i
			phi_hat_k[i] = phi_hat_global[global_i][0]

		patch_index += 1


def check_solution_bounded(patches):

	"""
	Check to see if the solution is bounded by seeing if 
	none of the values are nan
	"""

	for patch in patches:
		for  phi in patch.phi_hat:
			if math.isnan(phi):
				return False

	return True
