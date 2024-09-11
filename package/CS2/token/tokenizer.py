from typing import Union
from awpy import Demo
import pandas as pd
import numpy as np
import random

class Tokenizer:

    # Token Version (e. g. 100 for version 1.0.0)
    TOKEN_VERSION = '100'

    # Positions for encoding
    INFERNO_POSITIONS = [
        'a',
        'a_balcony',
        'aps',
        'arch',
        'b',
        'back_ally',
        'banana',
        'bridge',
        'ct_start',
        'deck',
        'graveyard',
        'kitchen',
        'library',
        'lower_mid',
        'mid',
        'pit',
        'quad',
        'ruins',
        'sec_mid',
        'sec_mid_balcony',
        't_aps',
        't_ramp',
        't_spawn',
        'top_mid',
        'under',
        'upstairs'
    ]


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Tokenization
    # --------------------------------------------------------------------------------------------

    def tokenize_match(self, df: pd.DataFrame, map_name: str, map_nodes: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenizes the given snapshots of the given dataframe.

        Parameters:
            - df: pd.DataFrame: The dataframe containing the snapshots to tokenize.
            - map: str: The name of the map. Can be one of the following: 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_vertigo', 'de_ancient', 'de_anubis'.
            - map_nodes: pd.DataFrame: The dataframe containing the graph nodes of the map.
        """

        # Validate the map name
        if map_name not in ['de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_vertigo', 'de_ancient', 'de_anubis']:
            raise ValueError(f"Invalid map name: {map_name}. The map name must be one of the following: 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_vertigo', 'de_ancient', 'de_anubis'.")

        # 1. Tokenize the positions
        df = self._TOKEN_positions_(df, map_name, map_nodes)

        # 2. Tokenize universal data
        df = self._TOKEN_universal_data_(df)

        return df
    


    # --------------------------------------------------------------------------------------------
    # REGION: Tokenization Private Functions
    # --------------------------------------------------------------------------------------------

    # 1. Tokenize the positions of the players
    def _TOKEN_positions_(self, df: pd.DataFrame, map_name: str, map_nodes: pd.DataFrame):
        """
        Encodes player positions in the given dataframe and returns the token.

        Parameters:
            - df: pd.DataFrame: The dataframe containing the snapshots to tokenize.
            - map: str: The name of the map. Can be one of the following: 'de_dust2', 'de_inferno', 'de_mirage', 'de_nuke', 'de_vertigo', 'de_ancient', 'de_anubis'.
            - map_nodes: pd.DataFrame: The dataframe containing the graph nodes of the map.
        """
        
        # Get all unique position names
        position_names = self.__INIT_get_position_names__(map_name)

        # Store new columns
        new_columns = {}

        # Add position names to each player's each snapshot in the dataframe
        for player_idx in range(0, 10):
            if player_idx < 5:
                new_columns[f'CT{player_idx}_pos_name'] = ''
            else:
                new_columns[f'T{player_idx}_pos_name'] = ''

        # Add new position-based player count columns
        for pos in position_names:
            new_columns[f"TOKEN_CT_POS_{pos}"] = 0
            new_columns[f"TOKEN_T_POS_{pos}"] = 0

        # Add the new columns to the dataframe
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

        # Add position names to each player's each snapshot in the dataframe
        for player_idx in range(0, 10):
            if player_idx < 5:
                df[f'CT{player_idx}_pos_name'] = df.apply(lambda x: self.__EXT_closest_node_pos_name__(x[f'CT{player_idx}_X'], x[f'CT{player_idx}_Y'], x[f'CT{player_idx}_Z'], map_nodes), axis=1)
            else:
                df[f'T{player_idx}_pos_name'] = df.apply(lambda x: self.__EXT_closest_node_pos_name__(x[f'T{player_idx}_X'], x[f'T{player_idx}_Y'], x[f'T{player_idx}_Z'], map_nodes), axis=1)

        # Set the position-based player count columns
        for pos in position_names:
            for player_idx in range(0, 10):
                if player_idx < 5:
                    df[f"TOKEN_CT_POS_{pos}"] += (df[f'CT{player_idx}_pos_name'] == pos).astype(int) * df[f'CT{player_idx}_is_alive'].astype(int)
                else:
                    df[f"TOKEN_T_POS_{pos}"] += (df[f'T{player_idx}_pos_name'] == pos).astype(int) * df[f'T{player_idx}_is_alive'].astype(int)

        # Create the CT and T token
        df['TOKEN_CT_POS'] = df[[f"TOKEN_CT_POS_{pos}" for pos in position_names]].astype(str).apply(lambda x: ''.join(x), axis=1)
        df['TOKEN_T_POS'] = df[[f"TOKEN_T_POS_{pos}" for pos in position_names]].astype(str).apply(lambda x: ''.join(x), axis=1)

        # Drop all new columns except the tokens
        df = df.drop(columns=new_columns.keys())

        return df
    


    # 2. Tokenize universal data
    def __EXT_set_CT_buy(self, row):

        # Economy values
        CT_economy = row['UNIVERSAL_CT_equipment_value']

        # Return the economy value
        if CT_economy < 5000:
            return 0
        elif CT_economy < 10000:
            return 1
        elif CT_economy < 15000:
            return 2
        else:
            return 3

    def __EXT_set_T_buy(self, row):

        # Economy values
        T_economy = row['UNIVERSAL_T_equipment_value']

        # Return the economy value
        if T_economy < 5000:
            return 0
        elif T_economy < 10000:
            return 1
        elif T_economy < 15000:
            return 2
        else:
            return 3

    def _TOKEN_universal_data_(self, df: pd.DataFrame):
        """
        Encodes universal data in the given dataframe and returns the token.

        Parameters:
            - df: pd.DataFrame: The dataframe containing the snapshots to tokenize.
        """

        # Store new columns
        new_columns = {}

        # Add new universal data columns
        new_columns['TOKEN_CT_BUY'] = 0
        new_columns['TOKEN_T_BUY'] = 0
        new_columns['TOKEN_CT_SCORE'] = 0
        new_columns['TOKEN_T_SCORE'] = 0
        new_columns['TOKEN_AFTERPLANT'] = 0
        new_columns['TOKEN_CT_WINS'] = 0


        # Add the new columns to the dataframe
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

        # Set the economy values
        df['TOKEN_CT_BUY'] = df.apply(self.__EXT_set_CT_buy, axis=1)
        df['TOKEN_T_BUY'] = df.apply(self.__EXT_set_T_buy, axis=1)

        # Set the score values
        df['TOKEN_CT_SCORE'] = df['UNIVERSAL_CT_score'].apply(lambda x: f'0{x}' if x < 10 else str(x))
        df['TOKEN_T_SCORE'] = df['UNIVERSAL_T_score'].apply(lambda x: f'0{x}' if x < 10 else str(x))

        # Store the site plant values
        plants_in_rounds = df[['UNIVERSAL_round', 'UNIVERSAL_is_bomb_planted_at_A_site', 'UNIVERSAL_is_bomb_planted_at_B_site']].copy()
        plants_in_rounds.drop_duplicates(subset=['UNIVERSAL_round'], keep='last', inplace=True)
        plants_in_rounds.rename(columns={'UNIVERSAL_is_bomb_planted_at_A_site': 'TOKEN_A_PLANT', 'UNIVERSAL_is_bomb_planted_at_B_site': 'TOKEN_B_PLANT'}, inplace=True)

        # Set the plant in the df by merging the plants_in_rounds
        df = df.merge(plants_in_rounds, on='UNIVERSAL_round', how='left')

        # Drop the bomb_site columns
        del plants_in_rounds

        # Set the afterplant flag values
        df['TOKEN_AFTERPLANT'] = df['UNIVERSAL_is_bomb_planted_at_A_site'].astype(int) + df['UNIVERSAL_is_bomb_planted_at_B_site'].astype(int)

        # Set the win values
        df['TOKEN_CT_WINS'] = df['UNIVERSAL_CT_wins']

        # Token version
        df['TOKEN_VERSION'] = self.TOKEN_VERSION

        # Create token by concatenating all the TOKEN columns as strings
        df['TOKEN'] = df['TOKEN_VERSION'].astype(str) + \
                      df['TOKEN_CT_POS'].astype(str) + \
                      df['TOKEN_T_POS'].astype(str) + \
                      df['TOKEN_CT_BUY'].astype(str) + \
                      df['TOKEN_T_BUY'].astype(str) + \
                      df['TOKEN_CT_SCORE'].astype(str) + \
                      df['TOKEN_T_SCORE'].astype(str) + \
                      df['TOKEN_A_PLANT'].astype(str) + \
                      df['TOKEN_B_PLANT'].astype(str) + \
                      df['TOKEN_AFTERPLANT'].astype(str) + \
                      df['TOKEN_CT_WINS'].astype(str)

        # Drop all columns starting with 'TOKEN_' except the token
        df = df.drop(columns=[col for col in df.columns if col.startswith('TOKEN_') and col != 'TOKEN'])

        return df



    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------

    # Calculate closest graph node to a position
    def __EXT_closest_node_pos_name__(self, coord_x, coord_y, coord_z, map_nodes):
        """
        Returns the closest node to a given position.
        
        Parameters:
        - coord_x: the x coordinate of the position.
        - coord_y: the y coordinate of the position.
        - coord_z: the z coordinate of the position.
        - nodes: the nodes dataframe.
        """

        distances = np.sqrt((map_nodes['X'] - coord_x)**2 + (map_nodes['Y'] - coord_y)**2 + (map_nodes['Z'] - coord_z)**2)
        return map_nodes.loc[distances.idxmin(), 'pos_name']
    
    # Get the position names for the given map
    def __INIT_get_position_names__(self, map):
        """
        Returns the position names for the given map.
        
        Parameters:
        - map: the name of the map.
        """

        if map == 'de_inferno':
            return self.INFERNO_POSITIONS
        else:
            print('WARNING: the selected map is under development, thus not usable yet. Please contact the developer for further information: random.developer@email.com.')
            return []