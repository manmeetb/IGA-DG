"""
Module: mesh.py
------------------------------------------

Generates the mesh for the given case
"""


import parameters
import patch

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


