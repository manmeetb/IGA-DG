"""
Module: simulation.py
------------------------------------------

Generates the simulation data structures and solves the required PDE
"""

import basis
import patch
import element
import plotter
import math
import numpy
import matplotlib.pyplot as plt
import solve_explicit
import parameters
import compute_error


def generate_mesh(reference_element):

	"""
	Generate the mesh of patches for the 1D domain.

	:param reference_element: The reference element for each patch on the mesh
	"""

	# Generate the patches that form the 1D mesh

	delta_x = float(parameters.CONST_X_RANGE[-1] - parameters.CONST_X_RANGE[0])/parameters.CONST_NUM_PATCH

	patches = []

	for i in range(parameters.CONST_NUM_PATCH):
		patch_x_range = [i*delta_x, (i+1)*delta_x]
		patches.append(patch.Patch(patch_x_range, reference_element, i))


	# Generate the patch faces. Impose periodic BCs here

	patch_faces = []
	
	for i in range(parameters.CONST_NUM_PATCH):
		patch_face = patch.PatchFace()

		if i == parameters.CONST_NUM_PATCH-1:
			# Periodic BC
			patch_face.set_patch_pointer(patches[i], 0)  # left face
			patch_face.set_patch_pointer(patches[0], 1)  # right face
		else:	
			patch_face.set_patch_pointer(patches[i], 0)  # left face
			patch_face.set_patch_pointer(patches[i+1], 1)  # right face

		patch_faces.append(patch_face)

	return patches, patch_faces


def set_initial_condition(patches):

	"""
	Set the initial condition on each patch by using a L2 projection

	:param patches: The list of patches that make up the mesh
	"""

	for patch in patches:
		patch.set_initial_condition(parameters.CONST_FUNCTION_IC)


def main():

	print "TEST: ML = %d, P = %d, NUM_PATCHES = %d " % (parameters.CONST_ML, parameters.CONST_P, 
		parameters.CONST_NUM_PATCH)

	# Generate the common reference element
	print "Generate Reference Element"
	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)

	if parameters.CONST_BASIS_TYPE == "SMOOTHED":
		knots_smooth = b_spline_basis.get_smoothed_knots()
		b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)

	reference_element = element.ReferenceElement(b_spline_basis)

	# Generate the "mesh" in the 1D domain, composed of patches
	print "Generate Patches/Mesh"
	patches, patch_faces = generate_mesh(reference_element)

	# Set the initial condition
	set_initial_condition(patches)

	# Solve the flow
	solve_explicit.solve_explicit(patches, patch_faces)

	# Compute the error and output the error file
	L2_error = compute_error.compute_L2_error(patches, parameters.CONST_FUNCTION_IC)
	compute_error.output_error_file(patches, L2_error)
	print "L2_error = %e " % (L2_error)

	# Plot the approximate solution
	if parameters.CONST_OUTPUT_PLOT:
		plotter.plot_numerical_solution(patches, plot_initial_condition=True)


if __name__ == "__main__":
	main()



