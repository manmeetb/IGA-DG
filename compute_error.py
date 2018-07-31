"""
Module: compute_error.py
------------------------------------------

Module used for computing the error values
"""

import integrate
import parameters

import math
import numpy
import matplotlib.pyplot as plt
import sys
import os
import os.path


def compute_L2_error(patches, f_exact):

	"""
	Compute the L2 error between the approximate solution and the exact
	solution.

	:param patches: The list of patches that form the mesh
	:param f_exact: The exact value of the function
	"""

	L2_Error = 0.0

	for patch in patches:

		# Obtain approximate solution on the physical domain
		u_h = lambda x, patch=patch: patch.get_phi(x)

		# Map the composite quadrature interval onto the physical domain
		composite_integral_break_pts_physical = [patch.mapping_xi_to_x(xi) for xi in 
			patch.reference_element.basis_ref_domain.composite_quadrature_break_points]

		# Number of quadrature nodes/weights. More points needed since
		# No longer integrating a polynomial
		quadrature_n = patch.reference_element.basis_ref_domain.p*4

		# Integration limits
		x_l, x_r = patch.x_vertices[0], patch.x_vertices[1]



		# print "composite_integral_break_pts_physical: " + str(composite_integral_break_pts_physical)
		# print "quadrature_n: " + str(quadrature_n)
		# print "x_l, x_r: %f, %f " % (x_l, x_r)

		# x_vals = numpy.linspace(x_l, x_r, 100)
		# y_vals = [u_h(x) for x in x_vals]
		# plt.plot(x_vals, y_vals, c='b')

		# y_vals = [f_exact(x) for x in x_vals]
		# plt.plot(x_vals, y_vals, c='r')

		# plt.grid()
		# plt.show(block=True)



		integrand_func = lambda x: (u_h(x) - f_exact(x))**2.0
		
		# x_vals = numpy.linspace(x_l, x_r, 200)
		# y_vals = [integrand_func(x) for x in x_vals]
		# plt.plot(x_vals, y_vals)
		# plt.show(block=True)
		# sys.exit(0)


		L2_Error += integrate.integrate_gauss_legendre_quadrature(integrand_func, quadrature_n, x_l, x_r, 
			interval_break_points=composite_integral_break_pts_physical)
		#L2_Error_mid_point += integrate.integrate_midpoint_rule(integrand_func, x_l, x_r, 2000)

	L2_Error =  math.sqrt(L2_Error)

	return L2_Error



def output_error_file(patches, L2_Error):

	"""
	Output the Error File for the given case
	"""

	# Get the total number of degrees of freedom
	n_dof = 0
	for patch in patches:
		n_dof += patch.n

	h = 1.0/n_dof

	# Get the file name. Set it using the specified prefix, ml and p
	ml = parameters.CONST_ML
	p = parameters.CONST_P

	file_name = "%s_ml%d_P%d.txt" % ("L2_Error", ml, p)
	file_abs_path = os.path.join(parameters.CONST_ERROR_OUTPUT_FILE_DIR, file_name)

	with open(file_abs_path, "w") as fp:

		# Print the header
		fp.write("N_DOF \t h \t ml \t p \t L2_Error \n")
		fp.write("%d \t %.14e \t %d \t %d \t %.14e \n" % (n_dof, h, ml, p, L2_Error))


