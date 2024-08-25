from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import pandas as pd


class NormalizePosition:

    # Position normalization constants
    POS_X_MIN = 0
    POS_X_MAX = 0
    POS_Y_MIN = 0
    POS_Y_MAX = 0
    POS_Z_MIN = 0
    POS_Z_MAX = 0

    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Normalize map graph
    # --------------------------------------------------------------------------------------------

    # Normalize the map graph nodes dataset
    def __PREP_validate_params__(self, df: pd.DataFrame, map_pos_dictionary: dict):

        # If the dataframe is None, throw an error
        if df is None:
            raise ValueError("The dataframe cannot be None.")
        
        # If the dataframe is not a pandas dataframe, throw an error
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The dataframe must be a pandas dataframe.")
        
        # If the dataframe doesn't have columns 'X', 'Y' and 'Z', throw an error
        if 'X' not in df.columns or 'Y' not in df.columns or 'Z' not in df.columns:
            raise ValueError("The dataframe must have columns 'X', 'Y' and 'Z'.")
        
        # If the map_norm_dict is None, throw an error
        if map_pos_dictionary is None:
            raise ValueError("The map_norm_dict cannot be None.")
        
        # If the map_norm_dict is not a dictionary, throw an error
        if not isinstance(map_pos_dictionary, dict):
            raise ValueError("The map_norm_dict must be a dictionary.")

    def __PREP_NORM_position_scaler__(self, map_pos_dictionary: dict):

        # If the map_norm_dict is not None, throw an error
        if map_pos_dictionary is None:
            raise ValueError("The map_norm_dict cannot be None.")
        
        # If the map_norm_dict is not a dictionary, throw an error
        if not isinstance(map_pos_dictionary, dict):
            raise ValueError("The map_norm_dict must be a dictionary.")
        
        # If the map_norm_dict doesn't have keys 'X', 'Y' and 'Z', throw an error
        if 'X' not in map_pos_dictionary or 'Y' not in map_pos_dictionary or 'Z' not in map_pos_dictionary:
            raise ValueError("The map_norm_dict must have keys 'X', 'Y' and 'Z'.")
        
        # Set the position normalization values
        self.POS_X_MIN = map_pos_dictionary['X']['min']
        self.POS_X_MAX = map_pos_dictionary['X']['max']
        self.POS_Y_MIN = map_pos_dictionary['Y']['min']
        self.POS_Y_MAX = map_pos_dictionary['Y']['max']
        self.POS_Z_MIN = map_pos_dictionary['Z']['min']
        self.POS_Z_MAX = map_pos_dictionary['Z']['max']
    
    def normalize(self, df: pd.DataFrame, map_pos_dictionary: dict) -> pd.DataFrame:

        # Validate the input dataframe
        self.__PREP_validate_params__(df, map_pos_dictionary)

        # Prepare the position normalization values
        self.__PREP_NORM_position_scaler__(map_pos_dictionary)

        # Transform the X, Y, Z columns
        df['X'] = (df['X'] - self.POS_X_MIN) / (self.POS_X_MAX - self.POS_X_MIN)
        df['Y'] = (df['Y'] - self.POS_Y_MIN) / (self.POS_Y_MAX - self.POS_Y_MIN)
        df['Z'] = (df['Z'] - self.POS_Z_MIN) / (self.POS_Z_MAX - self.POS_Z_MIN)

        return df


    # --------------------------------------------------------------------------------------------
    # REGION: DEPRECATED
    # --------------------------------------------------------------------------------------------


    # Normalize the map graph nodes dataset with scaler
    def normalize_with_scaler(
        self, 
        nodes, 
        pos_col_names=['X', 'Y', 'Z'], 
        scaler_operation='none', 
        scaler_save_path=None
    ):
        """
        DEPRACTED: Normalize the map node dataset's X, Y and Z coordinates.
        
        Parameters:
            - nodes: The nodes dataset of the map.
            - pos_col_names: The names of the positional columns to normalize. Default is ['X', 'Y', 'Z'].
            - scaler_operation: The operation to perform with the scaler. It can be 'save', 'return' or 'none'. Default is 'none'.
            - scaler_save_path: The path to which the scaler should be saved. Useful only if the scaler_operation is 'save'. Default is None.
        """

        # Check whether the filename ends with '.pkl'
        if scaler_save_path != None and scaler_save_path[-4:] != '.pkl':
            scaler_save_path += '.pkl'

        # Fit the scaler and transform the nodes dataset
        map_graph_scaler = MinMaxScaler()
        map_graph_scaler.fit(nodes[pos_col_names])
        nodes[pos_col_names] = map_graph_scaler.transform(nodes[pos_col_names])
        
        # If the scaler operation is 'save', save the model and return the nodes dataset
        if scaler_operation == 'save':

            if scaler_save_path == None:
                print('Path or filename was not declared, unable to save the scaler.')
                return nodes
            
            dump(map_graph_scaler, scaler_save_path)
            return nodes

        # If the scaler operation is 'return', return both the nodes dataset and the scaler
        elif scaler_operation == 'return':
            return nodes, map_graph_scaler
        
        # If the scaler operation is 'none', return only the nodes dataset
        else:
            return nodes
    
