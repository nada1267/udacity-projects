#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#                                                                             
# PROGRAMMER: nada ayman
# DATE CREATED:  25/10/2023                                
# REVISED DATE: 
# PURPOSE: Create the function get_pet_labels that creates the pet labels from 
#          the image's filename. This function inputs: 
#           - The Image Folder as image_dir within get_pet_labels function and 
#             as in_arg.dir for the function call within the main function. 
#          This function creates and returns the results dictionary as results_dic
#          within get_pet_labels function and as results within main. 
#          The results_dic dictionary has a 'key' that's the image filename and
#          a 'value' that's a list. This list will contain the following item
#          at index 0 : pet image label (string).
#
##
# Imports python modules
from os import listdir

# TODO 2: Define get_pet_labels function below please be certain to replace None
#       in the return statement with results_dic dictionary that you create 
#       with this function
# 
from os import listdir

def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames 
    of the image files. These pet image labels are used to check the accuracy 
    of the labels that are returned by the classifier function, since the 
    filenames of the images contain the true identity of the pet in the image.
    Be sure to format the pet labels so that they are in all lower case letters
    and with leading and trailing whitespace characters stripped from them.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a 
      List. The list contains for following item:
         index 0 = pet image label (string)
    """
    #make a file_list to include the right file name by handling with start with
    #list comprehension
    filename_list = [filename for filename in listdir(image_dir) if not filename.startswith('.')]
    #store name of labels
    pet_labels = []
    #store the results
    results_dic = {}
    #reformatting naming of files 
    for filename in filename_list:
      
        pet_label = ' '.join([word.strip() for word in filename.lower().split('_') if word.isalpha()])
        
        print('Filename =', filename, '    label =', pet_label)
        #store the pet label values in list withen iterations
        pet_labels.append(pet_label)
        #finally:display your results
        print('File:', filename)
        print('Empty dictionary has', len(results_dic), 'items')
        #fetch all files in results dic and ensure not repeating
        if filename not in results_dic:
            results_dic[filename] = [pet_label]
        else:
            print('WARNING: Key', filename, 'already exists in results_dic with value', results_dic[filename])
      #display all files and its labels
        
    print('Printing all key-value pairs in dictionary results_dic:')
    for key, value in results_dic.items():
        print('filename =', key, '    pet label =', value[0])
        
    print('Full dictionary has', len(results_dic), 'items')
    
    return results_dic