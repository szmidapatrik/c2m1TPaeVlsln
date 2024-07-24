from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
from joblib import dump, load
import pandas as pd
import os



class CS2_Tabular_Normalize:

    # Variables
    MATCH_LIST = []
    TABULAR_MATCHES_PATH = None
    OUTPUT_PATH = None


    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    # Normalize the map graph nodes dataset
    def normalize_map_graph(
        self, 
        nodes, 
        pos_col_names=['X', 'Y', 'Z'], 
        scaler_operation='none', 
        scaler_save_path=None, 
        scaler_save_name=None
    ):
        """
        Normalize the map node dataset's X, Y and Z coordinates. Parameters:
            - nodes: The nodes dataset of the map.
            - pos_col_names: The names of the positional columns to normalize. Default is ['X', 'Y', 'Z'].
            - scaler_operation: The operation to perform with the scaler. It can be 'save', 'return' or 'none'. Default is 'none'.
            - scaler_save_path: The path to which the scaler should be saved. Useful only if the scaler_operation is 'save'. Default is None.
            - scaler_save_name: The name as which the model will be saved. Useful only if the scaler_operation is 'save'. Default is None.
        """

        # Check whether the filename ends with '.pkl'
        if scaler_save_name != None and scaler_save_name[-4:] != '.pkl':
            scaler_save_name += '.pkl'

        # Fit the scaler and transform the nodes dataset
        map_graph_scaler = MinMaxScaler()
        map_graph_scaler.fit(nodes[pos_col_names])
        nodes[pos_col_names] = map_graph_scaler.transform(nodes[pos_col_names])
        
        # If the scaler operation is 'save', save the model and return the nodes dataset
        if scaler_operation == 'save':

            if scaler_save_path == None or scaler_save_name == None:
                print('Path or filename was not declared, unable to save the scaler.')
                return nodes
            
            else:
                dump(map_graph_scaler, scaler_save_path)
                return nodes

        # If the scaler operation is 'return', return both the nodes dataset and the scaler
        elif scaler_operation == 'return':
            return nodes, map_graph_scaler
        
        # If the scaler operation is 'none', return only the nodes dataset
        else:
            return nodes
    

    # Build scaling dictionary
    def build_scaling_dict(
        self, 
        folder_path,
        convention_type='prefix',
        convention_value=None,

    ):
        """
        Builds a dictionary of min and max values for each column in the dataset by reading the dictionary files of the given folder. Parameters:
            - folder_path: The path to the folder containing the dataset dinctionaries.
            - convention_type: The convention type to use for filtering the dictionary files. It can be 'prefix' or 'postfix'. Default is 'prefix'.
            - convention_value: The value to use for filtering the dictionary files. Default is None.
        """

        # Parameter validation
        if convention_type not in ['prefix', 'postfix']:
            print("Invalid convention type. Please use 'prefix' or 'postfix'.")
            return None

        # Read all files in the folder_path and filter the ones with the given postfix
        files = os.listdir(folder_path)

        if convention_type == 'prefix':
            files = [file for file in files if file[:len(convention_value)] == convention_value]
        elif convention_type == 'postfix':
            files = [file for file in files if file[-len(convention_value):] == convention_value]

        # Initialize the scaling dictionary
        scaling_dict = pd.read_csv(folder_path + files[0])

        # Iterate through the files and append the data to the scaling dictionary
        for file in files[1:]:

            temp_dict = pd.read_csv(folder_path + file)
            scaling_dict['other_min'] = temp_dict['min']
            scaling_dict['other_max'] = temp_dict['max']

            # Update 'min' and 'max' columns
            scaling_dict['min'] = scaling_dict[['min', 'other_min']].min(axis=1)
            scaling_dict['max'] = scaling_dict[['max', 'other_max']].max(axis=1)

            # Free up memory
            del temp_dict

        # Drop the 'other_min' and 'other_max' columns
        scaling_dict.drop(columns=['other_min', 'other_max'], inplace=True)




        # TODO: Build the dictionary independently of the player indexes (e.g. CT0_health and CT1_health should be scaled to the same range)




        return scaling_dict