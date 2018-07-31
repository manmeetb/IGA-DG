"""
Module: solve_explicit.py
------------------------------------------

Solve the PDE using an explicit solver
"""


import patch
import parameters
import math

def solve_explicit(patches, patch_faces):

	"""
	Solve the equations using an explicit scheme

	:param patches: The list of patches
	:param patch_faces: The list of patch faces
	"""

	time = 0.0

	for t_step in range(parameters.CONST_NUM_TIME_STEPS):

		# Perform the time step
		perform_time_step(patches, patch_faces)

		# Output Progress
		time += parameters.CONST_DELTA_T
		print "t_step: %d / %d, Time: %f " % (t_step+1, parameters.CONST_NUM_TIME_STEPS, time)


def compute_RHS(patches, patch_faces):

	"""
	Loop through the patches and solve the system of equations
	to obtain the derivative of the coefficients multiplying the
	basis functions, phi_hat (RHS)
	"""

	# 1) Set RHS to 0
	for patch in patches:
		patch.set_RHS_to_zero()


	# 2) Add RHS Face Contribution
	# Compute all numerical fluxes
	for patch_face in patch_faces:
		patch_face.compute_numerical_flux()

	# Compute RHS Face components
	for patch in patches:
		patch.add_RHS_face()


	# 3) Add RHS Volume Contribution
	for patch in patches:
		patch.add_RHS_volume()


	# 4) Solve system for the RHS (RHS of the first order ODE)
	for patch in patches:
		patch.compute_RHS()


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


def perform_time_step(patches, patch_faces):

	"""
	Loop through the patches and perform the time step to obtain the 
	updated coefficient values for the next time step
	"""

	if parameters.CONST_TIME_STEP_METHOD == "EULER":
		euler_time_step(patches, patch_faces)
	elif parameters.CONST_TIME_STEP_METHOD == "EXPLICIT_RK4_LOW_STORAGE":
		explicit_rk4_lowstorage_time_step(patches, patch_faces)


def euler_time_step(patches, patch_faces):

	"""
	Perform an Euler time step to update the solution
	"""

	# Compute the RHS for the first order system
	compute_RHS(patches, patch_faces)

	# Update the solution in each patch
	delta_t = parameters.CONST_DELTA_T

	for patch in patches:
		for i in range(patch.n):
			patch.phi_hat[i] += delta_t*patch.rhs[i][0]


def explicit_rk4_lowstorage_time_step(patches, patch_faces):

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
		compute_RHS(patches, patch_faces)

		for patch in patches:
			for i in range(patch.n):
				patch.phi_hat_p[i] *= rk4a[rk]
				patch.phi_hat_p[i] += delta_t * patch.rhs[i][0]
				patch.phi_hat[i]   += rk4b[rk] * patch.phi_hat_p[i]

	# Check that solution is still bounded
	check_solution_bounded(patches)




