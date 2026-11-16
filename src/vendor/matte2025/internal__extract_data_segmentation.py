# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:42:30 2023

@author: Arthur Matte
@contact: matte.arthur01[at]gmail.com
"""


#### OS related


import pathlib

def build_dictionary(input_folder):
    """
    Build a dictionary from a directory with vial labels as keys and lists of specimen folders as values.

    Parameters:
    - input_folder (str): The path to the input folder.

    Returns:
    - dict: A dictionary with vial labels as keys and lists of specimen folders as values.
    """
    dictionary = {}

    for file in pathlib.Path(input_folder).iterdir():
        # Iterate through all filenames in the folder
        obj = str(file)
        indiv_folder = obj[obj.rfind("\\") + 1:]  # Find the last "\\" in the string

        if "_z" in indiv_folder :
            indiv = indiv_folder[:indiv_folder.find("_z")]
        else: indiv = indiv_folder

        if indiv not in dictionary:
            dictionary[indiv] = [indiv_folder]
        else:
            dictionary[indiv] = dictionary[indiv] + [indiv_folder]

    for i in dictionary.keys():
        dictionary[i] = sorted(dictionary[i])  # Sort folders for consistency

    # Modify the dictionary to keep only the repeat scans when they exist
    for indiv in dictionary.keys():
        for i in dictionary[indiv]:
            if 'repeat' in i:
                rep = str(i)
                rep = rep[:rep.find('_repeat')]

                if rep in dictionary[indiv]:
                    index = dictionary[indiv].index(rep)
                    del dictionary[indiv][index]

    return dictionary



#### Excel related
def ID_row_to_process(data_sheet, col_of_label, col_of_status):
    """
    Identify rows to process based on the status column in a data sheet.

    Parameters:
    - data_sheet (openpyxl.worksheet.worksheet.Worksheet): The worksheet containing the data.
    - col_of_label (int): The column index of the label in the data sheet.
    - col_of_status (int): The column index of the status in the data sheet.

    Returns:
    - tuple: A tuple containing two lists - indiv_list and row_indiv_list.
      - indiv_list (list): List of labels for rows with a None status.
      - row_indiv_list (list): List of corresponding row indices.
    """
    # Initialize lists to store labels and row indices
    indiv_list = []
    row_indiv_list = []

    label_col = chr(ord('@') + col_of_label)
    status_col = chr(ord('@') + col_of_status)

    # Iterate through rows to identify rows with None status
    for i, status in enumerate(data_sheet[status_col]):
        if status.value is None:
            line = i + 1
            label = data_sheet[label_col + str(line)].value
            indiv_list.append(label)
            row_indiv_list.append(str(line))

    return indiv_list, row_indiv_list

import openpyxl

def extract_bv_and_cv_from_xlsx_file(xfile_name):
    """
    Extracts body volume (bv) and cuticle volume (cv) from an Excel (xlsx) file.

    Parameters:
    - xfile_name (str): The name of the Excel file.

    Returns:
    - tuple: A tuple containing two lists - body_volume and cuticle_volume.
    """
    # Open the Excel file
    xfile = openpyxl.load_workbook(xfile_name)

    # Select the first sheet as the place of interest
    sheet = xfile[xfile.sheetnames[0]]

    # Initialize lists to store body volume and cuticle volume
    body_volume = []
    cuticle_volume = []

    # Extract body volume from the first column (column 'A')
    bv_col = 1
    for row in sheet[chr(ord('@') + bv_col)]:
        if row.value is not None:
            body_volume.append(int(float(row.value)))

    # Extract cuticle volume from the second column (column 'B')
    cv_col = 2
    for row in sheet[chr(ord('@') + cv_col)]:
        if row.value is not None:
            cuticle_volume.append(int(float(row.value)))

    return body_volume, cuticle_volume





import os
import scipy.ndimage
import math

def compile_bv_and_cv(indiv, dic, input_folder):
    """
    Compile body volume (bv) and cuticle volume (cv) for an individual from multiple xlsx files.

    Parameters:
    - indiv (str): The individual for which to compile volumes.
    - dic (dict): A dictionary containing individual labels as keys and lists of specimen folders as values.
    - input_folder (str): The path to the input folder.

    Returns:
    - tuple: A tuple containing the compiled body volume and cuticle volume.
      - body_volume (float): The compiled body volume.
      - cuticle_volume (float): The compiled cuticle volume.
    """
    if len(dic[indiv]) == 1:
        # Case with a single xlsx file for the individual
        xfile_name = os.path.join(input_folder, list(dic[indiv])[0], f"{list(dic[indiv])[0]}.xlsx")
        body_volume_list, cuticle_volume_list = extract_bv_and_cv_from_xlsx_file(xfile_name)

        # Apply median filter
        filtered_body_volume = scipy.ndimage.median_filter(body_volume_list, size=25)
        filtered_cuticle_volume = scipy.ndimage.median_filter(cuticle_volume_list, size=25)

        # Sum volumes of each slice
        body_volume = sum(filtered_body_volume)
        cuticle_volume = sum(filtered_cuticle_volume)

    else :
        
        # Collect list of bv and cv from the various xlsx file of the individual
        ########################################################################
        
        body_volume_tot = []
        cuticle_volume_tot = []
        
        for file in dic[indiv] :
            xfile_name = os.path.join(input_folder, file, f"{file}.xlsx") # Setup the name of the xlsx file to look in
            body_volume, cuticle_volume = extract_bv_and_cv_from_xlsx_file(xfile_name) # Extract body volume and cuticle volume from the xlsx file
            
            if not body_volume == 'empty' : # code used when a folder is useless but should remain for the segmentation program
                body_volume_tot.append(body_volume)
                cuticle_volume_tot.append(cuticle_volume)
            
        # Find the overlap
        ##################  
        
        for k in range(len(body_volume_tot)-1, 0 , -1):    
            
            body_volume00 = body_volume_tot[k-1]
            body_volume01 = body_volume_tot[k]
            
            cuticle_volume00 = cuticle_volume_tot[k-1]
            cuticle_volume01 = cuticle_volume_tot[k]

            if len(body_volume00) <  10 or len(body_volume01) <  10 : body_volume = cuticle_volume = 'pb_xsls'; print('STOOOOOOOOOOOOOOOOP'); break
        
            else :
                
                diff = []
                overlap_list = []
                
                # This loop find the best overlap of values from the list of body volume and cuticle volume from the two folders
                for l in range (len(body_volume01)-10, 10, -1) :
                    
                    overlap = len(body_volume01) - l # overlap value
                    
                    contig_bv = body_volume01[l:]
                    contig_cv = cuticle_volume01[l:] 
                       
                    window_bv = body_volume00[ : overlap ]
                    window_cv = cuticle_volume00[ : overlap]
                    
                    # Loop comparing the diff between overlapping values
                    temp_diff = [abs(window_bv[d] - contig_bv[d]) + abs(window_cv[d] - contig_cv[d]) for d in range(len(window_bv))]
                        
                    diff.append( sum(temp_diff) / len(temp_diff) ) # get the average difference for the tested overlap
                    overlap_list.append(overlap)
                    
                    
                # plot to manually verify the accuracy
                # import matplotlib.pyplot as plt
                # # Plotting after the loop completes
                # plt.plot(overlap_list, diff)
                # plt.xlabel("Overlap")
                # plt.ylabel("Average Difference")
                # plt.title("Average Difference vs. Overlap")
                # plt.show()
                
                best_overlap = overlap_list[diff.index(min(diff))] # define best overlap as the index of the best alignement of values
                
                body_volume_tot[k-1] = body_volume01[ : len(body_volume01) - math.floor(best_overlap/2)] + body_volume00[ math.ceil(best_overlap/2) : ] #remplace inital list that will be take in the next round
                
                cuticle_volume_tot[k-1] = cuticle_volume01[ : len(cuticle_volume01) - math.floor(best_overlap/2)] + cuticle_volume00[ math.ceil(best_overlap/2) : ] #remplace inital list that will be take in the next round
        
        
        if not cuticle_volume == 'pb_xsls' :
        
            # apply a median filter on the list of value to fit continuity assumption in the values and so remove punctual errors        
            filtered_body_volume = scipy.ndimage.median_filter(body_volume_tot[0], size=25) 
            filtered_cuticle_volume = scipy.ndimage.median_filter(cuticle_volume_tot[0], size=25)
            
            # Sum final list to get the total body and cuticle volume
            body_volume = sum(filtered_body_volume)
            cuticle_volume = sum(filtered_cuticle_volume)

    return body_volume, cuticle_volume

   
#### Final function

from tqdm import tqdm

def extract_segmentation_results(input_folder, data_xfile_name,
                                body_volume_col, cuticle_volume_col,
                                col_of_label, col_of_status):
    """
    Extract and compile segmentation results and update a data xlsx file.

    Parameters:
    - input_folder (str): The path to the input folder containing segmentation data.
    - data_xfile_name (str): The name of the data xlsx file to be updated.
    - body_volume_col (int): The column index for writing body volume in the data xlsx file.
    - cuticle_volume_col (int): The column index for writing cuticle volume in the data xlsx file.
    - col_of_label (int): The column index for labels in the data xlsx file.
    - col_of_status (int): The column index for status in the data xlsx file.

    Returns:
    - None
    """
    # Build a dictionary containing individual numbers as keys and individual folders as values from the input directory
    dic = build_dictionary(input_folder)

    # Open the data xlsx file
    data_xfile = openpyxl.load_workbook(data_xfile_name)
    data_sheet = data_xfile[data_xfile.sheetnames[0]] # Attribute the first sheet as the place of interest

    # Identify individuals to retrieve data
    indiv_list, row_indiv_list = ID_row_to_process(data_sheet, col_of_label, col_of_status)

    # Process individuals
    for i in tqdm(range(len(indiv_list)), desc="Extract results…", ascii=False, ncols=75):
        indiv = indiv_list[i]

        if indiv in dic.keys():
            
            # Retrieve and compile the xlsx file containing data
            body_volume, cuticle_volume = compile_bv_and_cv(indiv, dic, input_folder)

            # Write results in the final xlsx file
            line = row_indiv_list[indiv_list.index(indiv)]

            place_data_bv = chr(ord('@') + body_volume_col) + str(line)
            data_sheet[place_data_bv] = body_volume

            place_data_cv = chr(ord('@') + cuticle_volume_col) + str(line)
            data_sheet[place_data_cv] = cuticle_volume

    data_xfile.save(data_xfile_name)  # Save the updated xlsx file






