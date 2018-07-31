"""
Module: plotter.py
------------------------------------------

Module used for plotting and visualizing results
"""


import numpy
import matplotlib.pyplot as plt
import parameters
import patch


def plot_numerical_solution(patches, plot_initial_condition=False):
	
	"""
	Loop through the patches and plot the numerical solution

	:param patches: The list of patches that make up the domain
	:param plot_initial_condition: (Optional) If true, will plot the exact 
		initial condition (not the L2 projection) with a dotted black line
	"""

	for patch in patches:

		x1, x2 = patch.x_vertices[0], patch.x_vertices[1]

		x_vals = numpy.linspace(x1, x2, parameters.CONST_NUM_PATCH_PLOT_PTS)
		y_vals = [patch.get_phi(x_val) for x_val in x_vals]

		plt.plot(x_vals, y_vals)

	
	# If specified, plot the exact initial condition as well
	if plot_initial_condition:
		num_plot_pts = len(patches) * parameters.CONST_NUM_PATCH_PLOT_PTS
		x_vals = numpy.linspace(parameters.CONST_X_RANGE[0], parameters.CONST_X_RANGE[1], 
			num_plot_pts)
		y_vals = [parameters.CONST_FUNCTION_IC(x_val) for x_val in x_vals]

		plt.plot(x_vals, y_vals, c='k', linestyle='--')

	plt.grid()
	plt.show(block=True)


