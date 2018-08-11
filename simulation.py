"""
Module: simulation.py
------------------------------------------

Generates the simulation data structures and solves the required PDE
"""

import basis
import element
import plotter
import math
import numpy
import matplotlib.pyplot as plt
import solve_explicit
import solve_explicit_global
import parameters
import compute_error
import mesh
import solution


def simulate():

	print "ML = %d, P = %d, NUM_PATCHES = %d, NUM_BASIS = %d " % (parameters.CONST_ML, parameters.CONST_P, 
		parameters.CONST_NUM_PATCH, parameters.CONST_NUM_BASIS)

	# Generate the common reference element
	print "Generate Reference Element"
	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)

	if parameters.CONST_BASIS_TYPE == "SMOOTHED":
		knots_smooth = b_spline_basis.get_smoothed_knots()
		b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)

	reference_element = element.ReferenceElement(b_spline_basis)

	# Generate the "mesh" in the 1D domain, composed of patches
	print "Generate Patches/Mesh"
	patches, patch_faces = mesh.generate_mesh(reference_element)

	# Set the initial condition
	solution.set_initial_condition(patches)

	# Solve the flow
	if parameters.CONST_SOLVER_TYPE == "STANDARD":
		solve_explicit.solve_explicit(patches, patch_faces)
	elif parameters.CONST_SOLVER_TYPE == "GLOBAL":
		solve_explicit_global.solve_explicit_global(patches, patch_faces)
	else:
		raise ValueError("Unknown Solver Type")

	# Compute the error and output the error file
	L2_error = compute_error.compute_L2_error(patches, parameters.CONST_FUNCTION_IC)
	compute_error.output_error_file(patches, L2_error)
	print "L2_error = %e " % (L2_error)

	# Plot the approximate solution
	if parameters.CONST_OUTPUT_PLOT:
		plotter.plot_numerical_solution(patches, plot_initial_condition=True)


if __name__ == "__main__":
	simulate()




