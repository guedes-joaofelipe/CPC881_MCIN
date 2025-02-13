import sys, os, argparse, multiprocessing
import papermill as pm
import numpy as np 
import pandas as pd
from utils import ProgressBar

def run_notebook(dict_parameters, algorithm):
    # print ("Running model tag {} plot ({} nodes) for the {} dataset".format(model_tag, args.nodes, dataset_tag))
    tag_dict_parameters = dict_parameters.copy()
    tag_dict_parameters.pop('input_data_filepath', None)
    tag_dict_parameters.pop('active', None)
    run_tag = '_'.join([str(x) for x in list(tag_dict_parameters.values())])
    # print ('Run tag: ', run_tag)

    output_folder = './Notebooks/Output_Notebooks/' + algorithm
    if not os.path.exists(output_folder):
        print ('Creating folder ' + output_folder)
        os.makedirs(output_folder)

    try:
        pm.execute_notebook(
            './Notebooks/run_{}.ipynb'.format(dict_parameters['algorithm']),
            os.path.join(output_folder, '[{}].ipynb'.format(run_tag)),
            parameters = dict_parameters
        )
    except Exception as e:        
        print ('Error running notebook with tag {}:\n{}'.format(run_tag, e))

if __name__ == '__main__':
    algorithm = 'cde'
    ############################################################################
    # All active registers
    # ##########################################################################    
    # df_parameters = pd.read_excel('./experiments_parameters.xlsx', sheet_name=algorithm)    
    # df_parameters.replace(to_replace={'None': None}, value=None, method=None, inplace=True)    
    # processes = list()    
    # for dict_parameters in df_parameters.to_dict(orient='records'):
    #     if dict_parameters['active']:      
    #         print (list(dict_parameters.values()))      
    #         p = multiprocessing.Process(target=run_notebook, args=(dict_parameters,algorithm))
    #         processes.append(p)
    #         p.start()    

    # for process in processes:
    #     process.join()

    # print ("End of experiments")
    ############################################################################
    # All active registers grouped by function IDs
    ############################################################################
    df_parameters = pd.read_excel('./experiments_parameters.xlsx', sheet_name=algorithm)    
    df_parameters.replace(to_replace={'None': None}, value=None, method=None, inplace=True)    
    df_parameters = df_parameters[df_parameters['active'] == True]
    for func_id in df_parameters['funcId'].unique():
        print ("\nProcessing Func ID " + str(func_id) + "\n")
        processes = list()    
        for dict_parameters in df_parameters[df_parameters['funcId'] == func_id].to_dict(orient='records'):            
            print (list(dict_parameters.values()))      
            p = multiprocessing.Process(target=run_notebook, args=(dict_parameters,algorithm))
            processes.append(p)
            p.start()    

        for process in processes:
            process.join()
# python papermill_plot_run.py -d=ML1M -n=10 -r=30 -f=5 -m=MostPopular,ItemKNN,NMF