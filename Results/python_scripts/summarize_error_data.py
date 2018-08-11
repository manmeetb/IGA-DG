
"""
Module: summarize_error_data.py
------------------------------------------

Output the plots of the error convergence to visualize the orders of convergence.
Output as well a table with the convergence order data

NOTE: For now, the *_p.txt file is read in the error folder
"""

import math
import numpy
import matplotlib.pyplot as plt
import sys
import os
from tabulate import tabulate

# =================================================================
# 					Input Parameters

# Different P Values to consider
CONST_P_VALUES = [1, 2, 3]

# The mesh levels for each P value (ith array corresponds to P[i])
CONST_ML_VALUES = [	[0, 1, 2, 3, 4],
					[0, 1, 2, 3, 4],
					[0, 1, 2, 3, 4],]

CONST_ERROR_FOLDER_ABS_PATH = "/Users/manmeetbhabra/Documents/McGill/Research/IGA-DG/Results/Convergence_Orders_Global_Solve/Tau0.5/Patch_Refinement/Smoothed_Knots"
CONST_TEST_CASE_PREFIX = "L2_Error" # Everything before the underscore before ml

CONST_PLOT_TITLE = "L2 Error Convergence (Smoothed Knots, Knot Refinement)"
CONST_Y_LABEL = "L2 Error"
CONST_X_LABEL = "h = 1/NDof"

# =================================================================


# The curve colors to be cycled through on each plot (plot the curves
# in order the same way for each plot so the curve colors are identical for each 
# P value for each plot.
CONST_CURVE_COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class TestCaseP(object):
		
	"""  
	Holds the data related to the given test case. All mesh level
	data for a given P value will be stored in this class
	"""

	def __init__(self, P):
			
		"""
		Initializes the TestCaseP class with the P value

		:param P: The order of test case
		"""

		self.P = P

		# Holds the data for each mesh level as a dictionary
		# of the form {h: value, L2_error: value, L2_error_theo: value}. Data stored in order from 
		# smallest to largest mesh level (but ordering won't matter since the data 
		# will be sorted according to mesh length (h)). 
		self.results = []


	def add_experimental_results(self, abs_file_path):

		"""
		Add the experimental results in the given file to the experimental_results
		array

		:param abs_file_path: The absolute path to the file with the data to read. 
			This should be the *_p.txt file

		:return: -
		"""

		with open(abs_file_path, "r") as fp:

			# Process the error file. Find which column holds the h and L2_Error data
			# and load it

			header_line = fp.readline().rstrip("\n")
			header_line_vals = header_line.split()
			
			h_data_index = -1
			L2_error_data_index = -1

			for column_title in header_line_vals:
				if column_title == "h":
					h_data_index = header_line_vals.index(column_title)
				elif column_title == "L2_Error":
					L2_error_data_index = header_line_vals.index(column_title)

			if h_data_index < 0 or L2_error_data_index < 0:
				raise ValueError("h or L2_Error data not found in error file : %s " % abs_file_path)

			data_line = fp.readline().rstrip("\n")
			data_line_vals = [float(x) for x in data_line.split()] 

			h_value = data_line_vals[h_data_index]
			L2_error_value = data_line_vals[L2_error_data_index]

			self.results.append({"L2_error": L2_error_value, "h": h_value})


	def compute_theoertical_results(self):

		"""
		Compute what the theoretical values for the convergence should be at each
		mesh level and store this data in a L2_error_theo key in the results list
		"""

		theoretical_conv_rate = self.P + 1

		# Sort according to mesh length
		self.results = sorted(self.results, key=lambda x: x['h'])

		ln_h_min = math.log(self.results[0]['h'])
		ln_L2_error_min = math.log(self.results[0]['L2_error'])
		self.results[0]['L2_error_theo'] = math.exp(ln_L2_error_min) # For h_min L2_error_theo is the same as s

		# Compute the L2_error_theo values
		for i in range(1, len(self.results)):

			ln_h_val = math.log(self.results[i]['h'])
			ln_L2_error_theo = theoretical_conv_rate*(ln_h_val - ln_h_min) + ln_L2_error_min

			self.results[i]['L2_error_theo'] = math.exp(ln_L2_error_theo)


	def plot_error_values(self, var_key, var_theo_key, curve_color):

		"""
		On the global plt object plot the curve for the error convergence for the
		given variable

		:param var_key: The key (string) for the variable whose error to plot
		:param var_theo_key: The key (string) for the variable's theoertical value
		:param curve_color: The color of the curve and theoretical curve
		"""

		x_vals = []
		y_vals = []
		y_vals_theo = []

		for data_val in self.results:

			x_vals.append(data_val['h'])
			y_vals.append(data_val[var_key])
			y_vals_theo.append(data_val[var_theo_key])

		curve_label = "P : " + str(self.P)
 
		plt.plot(x_vals, y_vals, c=curve_color, label=curve_label)
		plt.scatter(x_vals, y_vals, c=curve_color)
		plt.plot(x_vals, y_vals_theo, c=curve_color, linestyle="--")


def load_error_data():
	
	"""
	Load the error data from the error files in the provided directory

	:return: A list of TestCaseP objects holding the data for the error data
		for each P and their associated mesh levels
	"""

	TestCaseP_list = []

	for P in CONST_P_VALUES:
		# loop over all the P values
		TestCaseP_object = TestCaseP(P)

		for ml in CONST_ML_VALUES[CONST_P_VALUES.index(P)]:
			# loop over all the mesh levels

			error_file_name = CONST_TEST_CASE_PREFIX + "_ml" + str(ml) + "_P" + str(P) + ".txt"
			error_file_abs_path = os.path.join(CONST_ERROR_FOLDER_ABS_PATH, error_file_name)

			# load the error data
			TestCaseP_object.add_experimental_results(error_file_abs_path)

		# Get the theoretical error values now
		TestCaseP_object.compute_theoertical_results()
		TestCaseP_list.append(TestCaseP_object)
	
	return TestCaseP_list


def plot_error_convergence(TestCaseP_list):

	"""
	Plot the error convergence for the required variables

	:param TestCaseP_list: The list of test case objects
	"""

	# Create the entropy error plot
	for TestCaseP_obj in TestCaseP_list:

		curve_color = CONST_CURVE_COLORS[TestCaseP_list.index(TestCaseP_obj)]
		TestCaseP_obj.plot_error_values("L2_error", "L2_error_theo", curve_color)

	plt.gca().set_yscale('log')
	plt.gca().set_xscale('log')
	
	plt.title(CONST_PLOT_TITLE)
	plt.ylabel(CONST_Y_LABEL)
	plt.xlabel(CONST_X_LABEL)

	plt.legend()
	plt.grid()
	
	plt.show(block=True)


def output_table(TestCaseP_list):

	"""
	Output the convergence results in tabular format 

	:param TestCaseP_list: The list of test case P objects
	"""

	results_table = []

	# Header line
	results_table.append(["Polynomial", "h", "L2 Error", "Conv. Order"])

	for TestCaseP_obj in TestCaseP_list:

		test_case_error_data_table = []

		# Get the table of error values
		for data_val in TestCaseP_obj.results:

			test_case_error_data_table.append([
				data_val['h'],
				data_val['L2_error'],
				"-"
				])

		# reverse the order of the data with respect to the h (smallest h last)
		test_case_error_data_table = sorted(test_case_error_data_table, reverse=True, key=lambda x: x[0])

		# Compute the convergence orders
		for i in range(1, len(test_case_error_data_table)):

			L2_error_2 = test_case_error_data_table[i][1]
			L2_error_1 = test_case_error_data_table[i-1][1]

			h2 = test_case_error_data_table[i][0]
			h1 = test_case_error_data_table[i-1][0]

			conv_order = math.log(L2_error_2/L2_error_1)/ math.log(h2/h1)

			test_case_error_data_table[i][2] = conv_order

		
		# Add the rows to the table
		for i in range(len(test_case_error_data_table)):
			
			new_row = []
			if i == 0:
				new_row.append("P = %d " % TestCaseP_obj.P)
			else:
				new_row.append(" ")

			for val in test_case_error_data_table[i]:
				new_row.append(val)
			
			results_table.append(new_row)

		# Add an empty row between each P
		results_table.append([" ", " ", " ", " "])

	print tabulate(results_table)



def main():
	
	TestCaseP_list = load_error_data()

	output_table(TestCaseP_list)
	plot_error_convergence(TestCaseP_list)


if __name__ == "__main__":
	main()

