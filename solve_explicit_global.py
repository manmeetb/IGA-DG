
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

	Lh = numpy.zeros((size_Lh, size_Lh))

	# Add in the entries into the Lh matrix. 
	# NOTE: Assumes all elements have same number of dofs

	for patch in patches:

		patch_index = patches.index(patch)
		n = patch.n
		N_k = patch.N_k
		P_kMin1 = patch.P_kMin1

		# Fill the N_k matrix
		for i in range(n):
			for j in range(n):
				
				global_i = patch_index*n + i
				global_j = patch_index*n + j

				Lh[global_i][global_j] = N_k[i][j]


		# Fill the P_kMin1 matrix
		for i in range(n):
			for j in range(n):

				if patch_index == 0:
					
					# Periodic BCs
					global_i = (patch_index)*n + i
					global_j = (len(patches)-1)*n + j

				else:

					global_i = (patch_index)*n + i
					global_j = (patch_index-1)*n + j
				
				Lh[global_i][global_j] = P_kMin1[i][j]

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

