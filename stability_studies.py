"""
Module: stability_studies.py
------------------------------------------

Studies the stability properties of the linear advection scheme
"""

import basis
import element
import math
import cmath
import numpy
import matplotlib.pyplot as plt
import parameters
import solution
import mesh
import solve_explicit_global
import os.path


def filter_eigenvalues_into_modes(w_real_modes_lists, w_imag_modes_lists, w_vals, k_list, k_index):

	"""
	Filter the eigenvalues into the correct mode curves. A somewhat 
	heuristic way will be needed to do this. 

	We will essentially look at the approximation (a linear approximation) for what the next
	eigenvalue in the mode curve should be. Then, the eigenvalue that comes closest to 
	this linear approximation will be used for that mode curve.

	We will use the dispersion curves to figure out which eigenvalue is associated 
	with which mode. Then, knowing this, we can add to the dissipation curve as well


	:param w_real_modes_lists: The list holding the sublists of real parts for the eigenvalues
		for each mode. w_real_modes_lists = [[w_mode_1], [w_mode_2], ...]
	:param w_imag_modes_lists: The list holding the sublists of imaginary parts for the eigenvalues
		for each mode. w_imag_modes_lists = [[w_mode_1], [w_mode_2], ...]
	:param w_vals: The eigenvalues for this current k (wavenumber). 
	:param k_list: The list holding all the k values (used for the linear interpolation)
	:param k_index: The index for which wavenumber we are at in the list (k_list). Will only be able to
		use linear extrapolation when k_index > 1 (assuming 0 based indexing)
	"""

	if k_index == 0:
		
		# Do not worry about extrapolation and ordering, this is the first values 
		# to add to the list. Ordering will matter after this

		for w_val in w_vals:
			w_real_modes_lists[w_vals.index(w_val)].append(w_val.real)
			w_imag_modes_lists[w_vals.index(w_val)].append(w_val.imag)


	elif k_index == 1:
		
		# Still cannot do linear extrapolation. Therefore, put value into the list which has the
		# the smallest absolute distance between current value and previous value

		for w_val in w_vals:

			w_val_real = w_val.real
			w_val_imag = w_val.imag

			mode_index = None  # Mode index that this w_val will be associated with
			mode_index_w_real_min_abs_diff = None  # The minimum absolute difference
			
			for w_real_mode_list in w_real_modes_lists:

				if mode_index is None:
					# First mode being considered
					mode_index = w_real_modes_lists.index(w_real_mode_list)
					mode_index_w_real_min_abs_diff = abs(w_val_real - w_real_mode_list[-1])

				else:
					if abs(w_val_real - w_real_mode_list[-1]) < mode_index_w_real_min_abs_diff:
						mode_index_w_real_min_abs_diff = abs(w_val_real - w_real_mode_list[-1])
						mode_index = w_real_modes_lists.index(w_real_mode_list)

			# Place this value in the correct mode list
			w_real_modes_lists[mode_index].append(w_val_real)
			w_imag_modes_lists[mode_index].append(w_val_imag)

	else:

		# With more than 2 elements in the list, can now do linear extrapolation

		# Get the linearly extrapolated values
		linearly_extrapolated_w_real = []

		for w_real_mode_list in w_real_modes_lists:
			
			k3 = k_list[k_index]
			k2 = k_list[k_index-1]
			k1 = k_list[k_index-2]

			w2 = w_real_mode_list[-1]
			w1 = w_real_mode_list[-2]

			# Linearly extrapolated value
			w3_tilde = ((k3-k2)/(k2-k1))*(w2-w1) + w2
			linearly_extrapolated_w_real.append(w3_tilde)


		# Find the w_val (real part) that is closest to the extrapolated values
		for w_val in w_vals:
			
			w_val_real = w_val.real
			w_val_imag = w_val.imag

			mode_index = None  # Mode index that this w_val will be associated with
			mode_index_w_real_min_abs_diff = None  # The minimum absolute difference
			
			for w_val_real_extrapolated in linearly_extrapolated_w_real:

				if mode_index is None:
					# First mode being considered
					mode_index = linearly_extrapolated_w_real.index(w_val_real_extrapolated)
					mode_index_w_real_min_abs_diff = abs(w_val_real - w_val_real_extrapolated)

				else:
					if abs(w_val_real - w_val_real_extrapolated) < mode_index_w_real_min_abs_diff:
						mode_index_w_real_min_abs_diff = abs(w_val_real - w_val_real_extrapolated)
						mode_index = linearly_extrapolated_w_real.index(w_val_real_extrapolated)

			# Place this value in the correct mode list
			w_real_modes_lists[mode_index].append(w_val_real)
			w_imag_modes_lists[mode_index].append(w_val_imag)


def study_dispersion_dissipation():

	"""
	Study the dispersion and dissipation properties of the scheme
	"""

	# =================================
	#          Preprocessing
	# =================================

	print "Generate Basis Functions"
	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)
	knots_smooth = b_spline_basis.get_smoothed_knots()
	b_spline_basis_smooth = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)

	print "Generate Reference Element"
	reference_element = element.ReferenceElement(b_spline_basis)
	reference_element_smooth = element.ReferenceElement(b_spline_basis_smooth)

	print "Generate Mesh"
	patches, patch_faces = mesh.generate_mesh(reference_element)
	patches_smooth, patch_faces_smooth = mesh.generate_mesh(reference_element_smooth)


	# =================================
	#         Stability Study
	# =================================

	# Look at the first patch
	patch = patches[0]

	A, B, C = patch.A_k, patch.B_k, patch.C_k
	n_eig = A.shape[0]  # Number of eigenvalues

	h = patch.x_vertices[1] - patch.x_vertices[0]
	
	# go up till 63ish
	k_min = 0
	k_max = int(parameters.CONST_NUM_BASIS * math.pi * (1./h))
	num_k = 7*k_max
	k_vals = numpy.linspace(k_min, k_max, num_k)


	# Theoretical dispersion results (for only the physical mode)
	w_real_theo_physical_mode = [parameters.CONST_BETA * k for k in k_vals]  

	# Theoretical Dissipation results (for only the physical mode)
	w_imag_theo_physical_mode = [0.0 for k in k_vals]  

	# Dispersion results (all modes). Stores a list of lists
	# of the form [[w1], [w2], ...] where [w1] are the frequencies
	# of the first mode. 
	w_real_modes_lists = []
	for _ in range(n_eig):
		w_real_modes_lists.append([])

	
	# Dissipation results (all modes). Stored just line for the real 
	# part of the w values in real_w_modes
	w_imag_modes_lists = []
	for _ in range(n_eig):
		w_imag_modes_lists.append([])


	k_index = 0
	for k in k_vals:

		cos_kh = math.cos(k*h)
		sin_kh = math.sin(k*h)

		A_k = complex(cos_kh, -sin_kh) * A
		B_k = B
		C_k = complex(cos_kh, sin_kh) * C

		M = A_k + B_k + C_k

		# Get the eigenvalues
		l_eigenvalue = numpy.linalg.eigvals(M)

		# Get the w values (temporal frequency) for all modes
		w_vals = [ x * complex(0,1) for x in l_eigenvalue]

		filter_eigenvalues_into_modes(w_real_modes_lists, w_imag_modes_lists, w_vals, k_vals, k_index)
		k_index += 1


	# Normalize the dispersion data
	k_vals_hat = [k*float(h)/(parameters.CONST_NUM_BASIS) for k in k_vals]

	for i in range(len(w_real_modes_lists)):
		w_real_modes_lists[i] = [w_real*float(h)/(parameters.CONST_NUM_BASIS) for w_real in w_real_modes_lists[i]]
	w_real_theo_physical_mode = [w_real_theo*float(h)/parameters.CONST_NUM_BASIS for w_real_theo in w_real_theo_physical_mode]

	# Plot the dissipation and dispersion curves for each mode

	# Dispersion Curves	
	plt.figure(1)

	for w_real_mode_list in w_real_modes_lists:
		plt.plot(k_vals_hat, w_real_mode_list, marker='o', markersize=1)		
	plt.plot(k_vals_hat, w_real_theo_physical_mode, linestyle="--")
	
	plt.xlabel("k * h / n_dof_element")
	plt.ylabel("Real(w) * h / n_dof_element")
	
	plt.grid()
	plt.legend()

	
	# Dissipation Curves
	plt.figure(2)

	for w_imag_mode_list in w_imag_modes_lists:
		plt.plot(k_vals_hat, w_imag_mode_list)	
	plt.plot(k_vals_hat, w_imag_theo_physical_mode, linestyle="--")
	
	plt.xlabel("k")
	plt.ylabel("Imag(w)")

	plt.grid()
	plt.legend()
	

	plt.show(block=True)


	return

	# Output the data into a file
	output_file_dir = "Results/Dispersion_Dissipation_Results/Tau1.0/Uniform_Knots"

	P = parameters.CONST_P
	nBasis = parameters.CONST_NUM_BASIS

	file_name = "%s_P%d_NBasis%d" % ("DispersionDissipationResults", P, nBasis)
	file_rel_path = os.path.join(output_file_dir, file_name)

	with open(file_rel_path, "w") as fp:

		fp.write("num_modes %d \n" % (n_eig))
		fp.write("num_pts %d \n" % (len(k_vals_hat)))


		fp.write("Dispersion Data\n")

		for i in range(len(k_vals_hat)):
			
			fp.write("%.14e " % k_vals_hat[i])
			for w_real_mode_list in w_real_modes_lists:
				fp.write("%.14e " % w_real_mode_list[i])
			fp.write("\n")

		
		fp.write("Dissipation Data\n")

		for i in range(len(k_vals_hat)):
			
			fp.write("%.14e " % k_vals_hat[i])
			for w_imag_mode_list in w_imag_modes_lists:
				fp.write("%.14e " % w_imag_mode_list[i])
			fp.write("\n")


def study_spectra():

	"""
	Study the stability properties of the scheme by looking at the spectra
	"""

	# =================================
	#          Preprocessing
	# =================================

	print "Generate Basis Functions"
	b_spline_basis = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS)
	knots_smooth = b_spline_basis.get_smoothed_knots()
	b_spline_basis_smooth = basis.BSplineBasis(parameters.CONST_P, parameters.CONST_NUM_BASIS, knots_smooth)

	print "Generate Reference Element"
	reference_element = element.ReferenceElement(b_spline_basis)
	reference_element_smooth = element.ReferenceElement(b_spline_basis_smooth)

	print "Generate Mesh"
	patches, patch_faces = mesh.generate_mesh(reference_element)
	patches_smooth, patch_faces_smooth = mesh.generate_mesh(reference_element_smooth)

	print "Set Initial Condition"
	solution.set_initial_condition(patches)
	solution.set_initial_condition(patches_smooth)
	
	# =================================
	#         Stability Study
	# =================================

	# Discretization matrix
	Lh = solve_explicit_global.assemble_Lh(patches)
	Lh_smooth = solve_explicit_global.assemble_Lh(patches_smooth)
	
	w,v = numpy.linalg.eig(Lh)
	w_smooth, v_smooth = numpy.linalg.eig(Lh_smooth)

	w_real = numpy.real(w)
	w_imag = numpy.imag(w)

	w_smooth_real = numpy.real(w_smooth)
	w_smooth_imag = numpy.imag(w_smooth)

	plt.figure(1)
	plt.scatter(w_real, w_imag, c='b', label="Uniform Knots")
	plt.scatter(w_smooth_real, w_smooth_imag, c='r', label="Smoothed Knots")
	
	plt.xlabel("Real")
	plt.ylabel("Imaginary")
	
	plt.grid()
	plt.legend()

	plt.show(block=True)


	# =================================
	#          Postprocessing
	# =================================


if __name__ == "__main__":
	#study_spectra()
	study_dispersion_dissipation()


