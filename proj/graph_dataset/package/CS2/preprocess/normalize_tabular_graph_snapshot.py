from sklearn.preprocessing import MinMaxScaler
from joblib import dump
import pandas as pd


class NormalizeTabularGraphSnapshot:

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
    # REGION: Normalize the tabular snapshots
    # --------------------------------------------------------------------------------------------

    def noramlize(
        self,
        df: pd.DataFrame,
        dictionary: pd.DataFrame,
        map_pos_dictionary: dict,
    ):
        """
        Normalizes the dataset.
        
        Parameters:
            - df: the dataset to be normalized.
            - dictionary: the dictionary with the min and max values of each column.
            - map_pos_dictionary: the dictionary with the min and max values of the position columns.
        """

        # Setup the position scaler
        self.__PREP_NORM_position_scaler__(map_pos_dictionary)

        # Normalize position columns
        df = self.__NORMALIZE_positions__(df)

        # Normalize other columns
        for col in df.columns:
            
            # Format column name
            dict_column_name = col[3:] if col.startswith('CT') else col[2:] if (col.startswith('T') and col != 'TOKEN') else col

            # Skip columns that should not be normalized
            if self.__NORMALIZE_skip_column__(dict_column_name):
                continue

            # Normalize some special columns manually
            if self.__NORMALIZE_is_manual_normalize_column__(dict_column_name):
                df = self.__NORMALIZE_manual__(df, col)
                continue
            
            # Normalize the other columns
            else:

                col_min = dictionary.loc[dictionary['column'] == dict_column_name]['min'].values[0]
                col_max = dictionary.loc[dictionary['column'] == dict_column_name]['max'].values[0]

                if col_max == 0 and col_min == 0:
                    df[col] = 0
                else:
                    df[col] = (df[col] - col_min) / (col_max - col_min) 

        return df



    # --------------------------------------------------------------------------------------------
    # REGION: Normalization private methods
    # --------------------------------------------------------------------------------------------

    # Prepare position scaler
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

    # Normalize positions
    def __NORMALIZE_positions__(self, df: pd.DataFrame):

        for player_idx in range(0, 10):

            if player_idx < 5:
            
                # Transform the X, Y, Z columns
                df[f'CT{player_idx}_X'] = (df[f'CT{player_idx}_X'] - self.POS_X_MIN) / (self.POS_X_MAX - self.POS_X_MIN)
                df[f'CT{player_idx}_Y'] = (df[f'CT{player_idx}_Y'] - self.POS_Y_MIN) / (self.POS_Y_MAX - self.POS_Y_MIN)
                df[f'CT{player_idx}_Z'] = (df[f'CT{player_idx}_Z'] - self.POS_Z_MIN) / (self.POS_Z_MAX - self.POS_Z_MIN)

            else:

                # Transform the X, Y, Z columns
                df[f'T{player_idx}_X'] = (df[f'T{player_idx}_X'] - self.POS_X_MIN) / (self.POS_X_MAX - self.POS_X_MIN)
                df[f'T{player_idx}_Y'] = (df[f'T{player_idx}_Y'] - self.POS_Y_MIN) / (self.POS_Y_MAX - self.POS_Y_MIN)
                df[f'T{player_idx}_Z'] = (df[f'T{player_idx}_Z'] - self.POS_Z_MIN) / (self.POS_Z_MAX - self.POS_Z_MIN)
                

        # Normalize the bomb X, Y, Z columns
        df['UNIVERSAL_bomb_X'] = (df['UNIVERSAL_bomb_X'] - self.POS_X_MIN) / (self.POS_X_MAX - self.POS_X_MIN)
        df['UNIVERSAL_bomb_Y'] = (df['UNIVERSAL_bomb_Y'] - self.POS_Y_MIN) / (self.POS_Y_MAX - self.POS_Y_MIN)
        df['UNIVERSAL_bomb_Z'] = (df['UNIVERSAL_bomb_Z'] - self.POS_Z_MIN) / (self.POS_Z_MAX - self.POS_Z_MIN)

        # Set the bomb X, Y, Z columns to 0 if the bomb is not planted
        df.loc[df['UNIVERSAL_is_bomb_planted_at_A_site'] + df['UNIVERSAL_is_bomb_planted_at_B_site'] == 0, 'UNIVERSAL_bomb_X'] = 0
        df.loc[df['UNIVERSAL_is_bomb_planted_at_A_site'] + df['UNIVERSAL_is_bomb_planted_at_B_site'] == 0, 'UNIVERSAL_bomb_Y'] = 0
        df.loc[df['UNIVERSAL_is_bomb_planted_at_A_site'] + df['UNIVERSAL_is_bomb_planted_at_B_site'] == 0, 'UNIVERSAL_bomb_Z'] = 0

        return df

    # Check if the column should be skipped
    def __NORMALIZE_skip_column__(self, dict_column_name: str):

        # Skip the match_id, numerical_match_id, token, smokes and infernos active cols
        if dict_column_name in ['MATCH_ID', 'NUMERICAL_MATCH_ID', 'TOKEN', 'UNIVERSAL_tick']:
            return True

        # Skip the positional columns (already normalized with the position_scaler)
        if dict_column_name in ['_X', '_Y', '_Z', 'UNIVERSAL_bomb_X', 'UNIVERSAL_bomb_Y', 'UNIVERSAL_bomb_Z']:
            return True
        
        # Skip the name columns
        if dict_column_name == '_name' or dict_column_name == 'UNIVERSAL_CT_clan_name' or dict_column_name == 'UNIVERSAL_T_clan_name':
            return True

        # Skip the state-describing boolean columns (values are already 0 or 1)
        if dict_column_name.startswith('_is') or \
        dict_column_name.startswith('UNIVERSAL_is') or \
        dict_column_name.startswith('UNIVERSAL_bomb_mx_pos'):
            return True

        # Skip the inventory columns (values are already 0 or 1)
        if dict_column_name.startswith('_inventory'):
            return True

        # Skip the universal HLTV player stat columns (values are already normalized)
        if dict_column_name.startswith('_hltv'):
            return True

        # Skip the columns if the name includes '%' (values are already normalized)
        if '%' in dict_column_name:
            return True

        # Skip the active weapon flag columns (values are already 0 or 1)
        if dict_column_name.startswith('_active_weapon') and \
           dict_column_name not in ['_active_weapon_magazine_size', '_active_weapon_ammo', '_active_weapon_magazine_ammo_left_%', '_active_weapon_max_ammo', '_active_weapon_total_ammo_left_%',]:
            return True

        return False

    # Check if the column should be manually normalized
    def __NORMALIZE_is_manual_normalize_column__(self, dict_column_name: str):
        
        if dict_column_name in ['_health', '_armor_value', '_balance', 
                                'UNIVERSAL_round',  
                                'UNIVERSAL_CT_score', 'UNIVERSAL_T_score', 
                                'UNIVERSAL_CT_alive_num', 'UNIVERSAL_T_alive_num',
                                'UNIVERSAL_CT_total_hp', 'UNIVERSAL_T_total_hp',
                                'UNIVERSAL_CT_losing_streak', 'UNIVERSAL_T_losing_streak',]:
            return True
        
        return False

    # Normalize column manually        
    def __NORMALIZE_manual__(self, df: pd.DataFrame, column_name: str):

        # Normalize the health and armor value columns
        if '_health' in column_name or '_armor_value' in column_name:
            df[column_name] = df[column_name] / 100

        # Normalize the balance column
        if '_balance' in column_name:
            df[column_name] = df[column_name] / 16000

        # Normalize the round column
        if column_name == 'UNIVERSAL_round':
            df[column_name] = df[column_name] / 24

        # Normalize the CT and T score columns
        if column_name in ['UNIVERSAL_CT_score', 'UNIVERSAL_T_score']:
            df[column_name] = df[column_name] / 12

        # Normalize the CT and T alive number columns
        if column_name in ['UNIVERSAL_CT_alive_num', 'UNIVERSAL_T_alive_num']:
            df[column_name] = df[column_name] / 5

        # Normalize the CT and T total HP columns
        if column_name in ['UNIVERSAL_CT_total_hp', 'UNIVERSAL_T_total_hp']:
            df[column_name] = df[column_name] / 500

        # Normalize the CT and T losing streak columns
        if column_name in ['UNIVERSAL_CT_losing_streak', 'UNIVERSAL_T_losing_streak']:
            df[column_name] = df[column_name] / 5

        return df
    
    