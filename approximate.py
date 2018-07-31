"""
Module: approximate.py
------------------------------------------

Use an L2 projection to approximate a given function using a given set of spline
functions over some given interval.
"""

import numpy
import matplotlib.pyplot as plt
import math
import sys

import integrate
import basis



def compute_L2_projection_M_matrix(function_exact, function_space_Vh, domain, quadrature_n, 
	interval_break_points=None):
	
	"""
	Compute the Mass matrix for the L2 projection linear solve
	"""

	# Form the Mass matrix. To make the computation efficient,
	# exploit the symmetric nature of M
	n = len(function_space_Vh)
	M = numpy.zeros((n, n))

	for i in range(n):
		for j in range(i+1):

			func = lambda x, i=i, j=j: function_space_Vh[i](x) * function_space_Vh[j](x)
			M[i][j] = integrate.integrate_gauss_legendre_quadrature(func, quadrature_n, domain[0], domain[1], 
				interval_break_points=interval_break_points)

			M[j][i] = M[i][j]

	return M


def compute_L2_projection_b_matrix(function_exact, function_space_Vh, domain, quadrature_n, 
	interval_break_points=None):
	
	"""
	Compute the load matrix (or more precisely vector) for the L2 projection linear solve
	"""

	# Form the Mass matrix. To make the computation efficient,
	# exploit the symmetric nature of M
	n = len(function_space_Vh)
	b = numpy.zeros((n,1))

	for i in range(n):
		func = lambda x, i=i: function_space_Vh[i](x) * function_exact(x)
		b[i][0] = integrate.integrate_gauss_legendre_quadrature(func, quadrature_n, domain[0], domain[1], 
				interval_break_points=interval_break_points)

	return b


def perform_L2_projection(function_exact, function_space_Vh, domain, quadrature_n, 
	interval_break_points=None, M=None, b=None):

	"""
	Perform an L2 projection of a given function, function_exact, over a given domain 
	using a set of functions in the function space function_space_Vh. This function handles
	L2 projections over 1D domains only.

	:param function_exact: The exact function
	:param function_space_Vh: The space of functions to use for the approximation. The
		functions are placed in a list
	:param domain: The list holding the domain over which to approximate the function [a,b]
	:param quadrature_n: The number of nodes used for the quadrature
	:param interval_break_points: (Optional) Specify subintervals over which to integrate 
		when performing the projection when using a composite gauss legendre quadrature.

	:return: A tuple of the form (list, approximate_function). The list corresponds to the list of 
		coefficients that, used in a linear combination with the functions in function space Vh, 
		form the approximation. approximate_function is the lambda expression for the function 
		approximation.
	"""

	# Form the Mass matrix (if needed)
	if M is None:
		M = compute_L2_projection_M_matrix(function_exact, function_space_Vh, domain, quadrature_n, 
			interval_break_points=interval_break_points)

	# Form the load vector (b)
	if b is None:
		b = compute_L2_projection_b_matrix(function_exact, function_space_Vh, domain, quadrature_n, 
				interval_break_points=interval_break_points)

	# Perform the linear solve
	coefficients = numpy.linalg.solve(M, b)

	f_approx = lambda x: basis.approximation_function(coefficients, function_space_Vh, x)

	return coefficients, f_approx


def compute_L2_error(function_exact, function_approx, domain, quadrature_n, interval_break_points=None):

	"""
	Compute the L2 error of the difference between the exact function and its 
	approximation over the domain [a,b].

	:param function_exact: The lambda expression for the exact function
	:param function_approx: The lambda expression for the function to approximate
	:param domain: The list containing the endpoints of the domain [a,b]
	:param quadrature_n: The number of nodes and weights to use for the quadrature.
	:param interval_break_points: The list holding the 
	
	:return : The value of the L2 error. Computed as 
		integral{from a to b} ( (function_exact - function_approx)^2 )
	"""

	f_diff = lambda x: (function_exact(x) - function_approx(x))**2

	L2_error = integrate.integrate_gauss_legendre_quadrature(f_diff, quadrature_n, domain[0], domain[1], 
				interval_break_points=interval_break_points)

	return L2_error


"""
=================================================
		Approximation testing functions
=================================================
"""


def add_function_to_plot(f, domain, label=None, color=None):

	"""
	Add the function to the given plot

	:param f: The function to be added to the plot
	:param domain: The list with the endpoints of the domain ([a,b])
	"""

	x_vals = numpy.linspace(domain[0], domain[1], 400)
	y_vals = []

	for x in x_vals:
		y_vals.append(f(x))

	plt.plot(x_vals, y_vals, c=color, label=label)


def test_oscillatory_approximation():

	# Output the results into a text file
	results_file_name = "Results/L2_errors_cosine_approximation_smooth_knots"

	# Get the approximation function space (B Spline Space) and the smoothed knot B Spline space
	spline_p = 4
	spline_n = 20
	b_spline_basis = basis.BSplineBasis(spline_p, spline_n)
	knots_smooth = b_spline_basis.get_smoothed_knots()
	b_spline_basis_smooth = basis.BSplineBasis(spline_p, spline_n, knot_vector=knots_smooth)


	# The basis functions to use for the approximation and interval break points for the
	# composite gaussian quadrature
	basis_funcs = b_spline_basis_smooth.basis_functions
	basis_interval_break_pts = b_spline_basis_smooth.knot_vector_unique


	L2_error_values_history = []

	# Loop over the different wave numbers
	
	i_min = 1
	i_max = int(spline_n**2)

	for i in range(i_min, i_max):

		k = float(i)/spline_n
		f_exact = lambda x, k=k: math.cos( (2.0*k-1.0) * math.pi * 0.5 * x)

		if i == i_min:
			
			# The mass matrix is constant for all approximations so do not compute it 
			# multiple times

			M = compute_L2_projection_M_matrix(f_exact, basis_funcs, 
				[-1, 1], spline_p+1, interval_break_points=basis_interval_break_pts)


		coefficients, f_approx = perform_L2_projection(f_exact, basis_funcs, 
			[-1, 1], spline_p+1, interval_break_points=basis_interval_break_pts, M=M)

		L2_error_values_history.append( 
			(k, compute_L2_error(f_exact, f_approx, [-1., 1.], spline_p+1, 
				interval_break_points=b_spline_basis.knot_vector_unique))
		)

		print "wave_number_progress : %d/%d, L2_error: %e " % (i, i_max, L2_error_values_history[-1][1])


	with open(results_file_name, "w") as fp:

		for L2_error_value_tuple in L2_error_values_history:
			fp.write("%.14e %.14e\n" % (L2_error_value_tuple[0], L2_error_value_tuple[1]))


def test_oscillatory_approximation_comparison():

	"""
	Approximate the given approximate function and plot the results to visualize
	"""

	# Get the approximation function space (B Spline Space) and the smoothed knot B Spline space
	spline_p = 4
	spline_n = 20
	b_spline_basis = basis.BSplineBasis(spline_p, spline_n)
	knots_smooth = b_spline_basis.get_smoothed_knots()
	b_spline_basis_smooth = basis.BSplineBasis(spline_p, spline_n, knot_vector=knots_smooth)

	k = 7
	f_exact = lambda x: math.cos( (2.0*k-1.0) * math.pi * 0.5 * x)

	coefficients, f_approx_uniforn = perform_L2_projection(f_exact, b_spline_basis.basis_functions, 
		[-1, 1], spline_p+1, interval_break_points=b_spline_basis.knot_vector_unique)
	print "Approximated Uniform"

	coefficients, f_approx_smooth = perform_L2_projection(f_exact, b_spline_basis_smooth.basis_functions, 
		[-1, 1], spline_p+1, interval_break_points=b_spline_basis_smooth.knot_vector_unique)	
	print "Approximated Smooth"
	
	plt.figure(1)
	add_function_to_plot(f_exact, [-1, 1], label="Exact", color="C2")
	add_function_to_plot(f_approx_uniforn, [-1, 1], label="Uniform Knots", color="C0")
	add_function_to_plot(f_approx_smooth, [-1, 1], label="Smooth Knots", color="C1")

	plt.grid()
	plt.gca().set_ylim(-1.6, 1.2)
	leg = plt.legend()
	leg.draggable()
	plt.title("Wave Number k = 7")
	plt.show(block=True)


def plot_L2_error_function_approximation():

	"""
	Function used to plot the L2 error variation of the function approximation 
	as a function of the wave number.
	"""

	# Holds tuples of the form 
	# 	(rel_path_to_file, legend_key)

	file_list = [
		("Results/L2_errors_cosine_approximation_uniform_knots", "Uniform Knots"),
		("Results/L2_errors_cosine_approximation_smooth_knots", "Smooth Knots"),
		("Results/L2_errors_cosine_approximation_uniform_knots_bquad4p", "Uniform Knots, higher quad"),
		("Results/L2_errors_cosine_approximation_smooth_knots_bquad4p", "Smooth Knots, higher quad")
	]


	file_data_points = []

	for file_data_tuple in file_list:

		pt_list = []
		
		with open(file_data_tuple[0], "r") as fp:
			while True:
				line = fp.readline()
				
				if line == "":
					break

				pt = [float(x) for x in line.split()]
				pt_list.append(pt)

		file_data_points.append(pt_list)

	# Plot the values
	plt.figure(1)
	for file_data in file_data_points:

		x_vals = [p[0] for p in file_data]
		y_vals = [p[1] for p in file_data]

		label = file_list[file_data_points.index(file_data)][1]
		plt.plot(x_vals, y_vals, label=label)

	plt.grid()
	plt.xlabel("wave number (k)")
	plt.ylabel("L2 Error")
	plt.legend()

	plt.figure(2)
	for file_data in file_data_points:

		x_vals = [p[0] for p in file_data]
		y_vals = [p[1] for p in file_data]

		label = file_list[file_data_points.index(file_data)][1]
		plt.semilogy(x_vals, y_vals, label=label)

	plt.grid()
	plt.gca().set_ylim(10E-14, 10)
	plt.xlabel("wave number (k)")
	plt.ylabel("L2 Error")
	plt.legend()	

	plt.show(block=True)





if __name__ == "__main__":

	if len(sys.argv) != 2:
		raise ValueError("Insufficient Number of Command Line Arguments")

	if sys.argv[1] == "plot":
		plot_L2_error_function_approximation()
	elif sys.argv[1] == "approximate":
		test_oscillatory_approximation()
	elif sys.argv[1] == "approximate_comparison":
		test_oscillatory_approximation_comparison()
	else:
		raise ValueError("Unknown Command Line Argument")




