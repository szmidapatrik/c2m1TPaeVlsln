from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
from joblib import dump, load
import pandas as pd
import os



class CS2_Dictionary:

    # Variables
    MATCH_LIST = []
    TABULAR_MATCHES_PATH = None
    OUTPUT_PATH = None


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Public methods - Build dictionary
    # --------------------------------------------------------------------------------------------

    # Build scaling dictionary
    def build_dictionary(
        self, 
        folder_path,
        convention_type='prefix',
        convention_value=None,

    ):
        """
        Builds a dictionary of min and max values for each column in the dataset by reading the dictionary files of the given folder.
        
        Parameters:
            - folder_path: str: The path to the folder containing the dictionary files.
            - convention_type: str: The convention type used in the dictionary files. It can be 'prefix' or 'postfix'. Default is 'prefix'.
            - convention_value: str: The convention value used in the dictionary files. Default is None
        """

        # Parameter validation
        if convention_type not in ['prefix', 'postfix']:
            print("Invalid convention type. Please use 'prefix' or 'postfix'.")
            return None

        # 1. Build the player-variant scaling dictionary
        scaling_dict = self.__scaling_dict_1__(folder_path, convention_type, convention_value)

        # 2. Make it player-invariant
        scaling_dict = self.__scaling_dict_player_invariant__(scaling_dict)

        return  scaling_dict
    
    # Build scaling dictionary for a single match
    def build_single_dictionary(
        self, 
        dictionary: pd.DataFrame,
    ):
        """
        Builds a dictionary of min and max values for each column in the dataset for a single match.
        
        Parameters:
            - dictionary: pd.DataFrame: the initial dictionary.
        """

        return self.__scaling_dict_player_invariant__(dictionary)



    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------

    # 1. Create the player-variant scaling dictionary
    def __scaling_dict_player_variant__(self, folder_path, convention_type, convention_value):
        
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

        return scaling_dict

    # 2. Create the player-invariant scaling dictionary
    def __scaling_dict_player_invariant__(self, scaling_dict: pd.DataFrame):

        # Initialize the player_column_prefix list, player_columns dictionary and player_dict DataFrame
        player_column_prefix = ['CT0', 'CT1', 'CT2', 'CT3', 'CT4', 'T5', 'T6', 'T7', 'T8', 'T9']
        player_columns = {}
        player_dict = pd.DataFrame()

        for prefix in player_column_prefix:

            # Filter the scaling_dict for the current prefix    
            player_columns[prefix] = scaling_dict[scaling_dict['column'].str.startswith(prefix)]

            # If it is the first prefix, copy the database and remove the prefix from the 'column' column
            if prefix == 'CT0':
                player_dict = player_columns[prefix].copy()
                player_dict['column'] = player_dict['column'].apply(lambda x: x.replace(prefix, ''))
                continue

            # If it is not the first prefix, concat the current prefix database with the dictionary
            else:
                temp = player_columns[prefix][['min', 'max']].rename(columns={'min': 'other_min', 'max': 'other_max'}).reset_index(drop=True).copy()
                player_dict = pd.concat([player_dict, temp], axis=1)
                del temp


            # Update the 'min' and 'max' columns
            player_dict['min'] = player_dict[['min', 'other_min']].min(axis=1)
            player_dict['max'] = player_dict[['max', 'other_max']].max(axis=1)

            # Drop the 'other_player_min' and 'other_player_max' columns
            player_dict.drop(columns=['other_min', 'other_max'], inplace=True)

        # Flter the scaling_dict for non_player columns and append the player_column_dict to the scaling_dict
        non_player_columns = scaling_dict[scaling_dict['column'].str.startswith('UNIVERSAL_')]
        scaling_dict = pd.concat([non_player_columns, player_dict], axis=0)

        # Free up memory
        del player_columns, player_dict, non_player_columns

        return scaling_dict