from sklearn.preprocessing import MinMaxScaler
from IPython.display import clear_output
from joblib import dump, load
import pandas as pd



class CS2_Tabular_Normalize:

    # Variables
    MATCH_LIST = []
    TABULAR_MATCHES_PATH = None
    OUTPUT_PATH = None



    # ---------- POSITIONAL COLUMNS ----------

    # Player positional columns
    PLAYER_POS_NORM_COLS = [['CT-NUMBER-_X', 'CT-NUMBER-_Y', 'CT-NUMBER-_Z'], ['T-NUMBER-_X', 'T-NUMBER-_Y', 'T-NUMBER-_Z']]

    # Overall positional columns, which are normalized with the min-max scaler of the inferno graph nodes
    OVERALL_POS_NORM_COLS = ['bomb_X', 'bomb_Y', 'bomb_Z']



    # ---------- NON-POSITIONAL COLUMNS ----------

    # Other player columns to normalize
    PLAYER_NORM_COLS = ['playerNUMBER_z', 'playerNUMBER_eyeZ', 'playerNUMBER_velocityX', 'playerNUMBER_velocityY', 'playerNUMBER_velocityZ',
                        'playerNUMBER_hp', 'playerNUMBER_armor',
                        'playerNUMBER_flashGrenades', 'playerNUMBER_smokeGrenades', 'playerNUMBER_heGrenades', 'playerNUMBER_totalUtility',
                        'playerNUMBER_equipmentValue', 'playerNUMBER_equipmentValueRoundStart',
                        'playerNUMBER_stat_kills', 'playerNUMBER_stat_HSK', 'playerNUMBER_stat_openKills', 'playerNUMBER_stat_tradeKills', 'playerNUMBER_stat_deaths', 'playerNUMBER_stat_openDeaths',
                        'playerNUMBER_stat_assists', 'playerNUMBER_stat_flashAssists', 'playerNUMBER_stat_damage', 'playerNUMBER_stat_weaponDamage', 'playerNUMBER_stat_nadeDamage']

    # Overall columns to normalize
    OVERALL_NORM_COLS = ['roundNum', 'tScore', 'ctScore', 'endTScore', 'endCTScore',
        'CT_aliveNum', 'T_aliveNum', 'CT_equipmentValue', 'T_equipmentValue',
        'bomb_Z', 'time_remaining', 'CT_totalHP', 'T_totalHP']



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    # Normalize the Inferno graph nodes dataset
    def normalize_map_graph(self, nodes, pos_col_names=['X', 'Y', 'Z'], scaler_operation='save', scaler_save_path=None, scaler_save_name=None):
        """
        Normalize the map node dataset's X, Y and Z coordinates. Parameters:
            - nodes: The nodes dataset of the map.
            - pos_col_names: The names of the positional columns to normalize. Default is ['X', 'Y', 'Z'].
            - scaler_operation: The operation to perform with the scaler. It can be 'save' or 'return'. Default is 'save'.
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
    


    # Collective normalization of the datasets
    def collective_normalization(self, dataset_folder_path, match_list, map_graph_scaler, output_folder_path=None):
        """
        Normalize the datasets collectively. Parameters:
            - dataset_folder_path: The folder path of the datasets.
            - match_list: The list of the dataset names.
            - map_graph_scaler: The map graph scaler.
            - output_folder_path: The folder path of the output datasets.
        """

        # If the output folder path is not provided, use the default one
        if output_folder_path == None:
            output_folder_path = dataset_folder_path

        # Get the min-max values of all datasets
        player_min_max_values, overall_min_max_values = self.__build_scaling_dictionary__(dataset_folder_path, match_list)
        print('Player and Overall min-max values are calculated.')


        # Iterate through the datasets
        for df_idx, df_name in enumerate(match_list):
            df = pd.read_csv(dataset_folder_path + df_name)



            # -------------------------
            # Normalize overall columns
            # -------------------------

            # Normalize overall positional columns
            df[self.OVERALL_POS_NORM_COLS] = map_graph_scaler.transform(df[self.OVERALL_POS_NORM_COLS].rename(columns={'bomb_X': 'x', 'bomb_Y': 'y'}))

            # Normalize overall columns
            for col in self.OVERALL_NORM_COLS:
                df[col] = (df[col] - overall_min_max_values[col + "_min"]) / (overall_min_max_values[col + "_max"] - overall_min_max_values[col + "_min"])




            # -------------------------
            # Normalize player columns
            # -------------------------

            # Normalize the player columns
            for player_idx in range(0, 10):

                # Normalize positional columns:  [pos_x, pos_y], [eye_x, eye_y]
                for pos_col_pair in self.PLAYER_POS_NORM_COLS:
                    updated_pos_col_pair = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in pos_col_pair]
                    df[updated_pos_col_pair] = map_graph_scaler.transform(df[updated_pos_col_pair].rename(columns={'player{}_x'.format(player_idx): 'x',
                                                                                                                   'player{}_y'.format(player_idx): 'y',
                                                                                                                   'player{}_eyeX'.format(player_idx): 'x',
                                                                                                                   'player{}_eyeY'.format(player_idx): 'y'}))

                # Normalize non-positional columns
                updated_norm_cols = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in self.PLAYER_NORM_COLS]
                for col_iter_idx, col in enumerate(updated_norm_cols):
                    df[col] = (df[col] - player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"]) / (player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] - player_min_max_values[player_norm_cols[col_iter_idx] + "_min"])
            


            # Save the normalized dataset
            df.to_csv(output_folder_path + f'norm_{df_name}', index=False)

            # Print the progress
            clear_output(wait=True)
            progress_percentage = (df_idx + 1) / len(match_list) * 100
            print(f'{df_idx + 1}/{len(match_list)} ({progress_percentage:.1f}%) datasets are normalized.')
            print('[' + '#' * int(progress_percentage / 10) + '·' * (10 - int(progress_percentage / 10)) + ']')


    
    # Match normalization with the map scaler model
    def match_normalization(self, match_dataset, map_graph_scaler: MinMaxScaler):
        """
        Normalize one match dataset. Parameters:
            - match_dataset: The match dataset.
            - map_graph_scaler: The map graph scaler.
        """

        # Get the min-max values of all datasets
        player_min_max_values, overall_min_max_values = self.__build_scaling_dictionary__(match_dataset)

        # Create new variable for the same dataset
        df = match_dataset

        # -------------------------
        # Normalize overall columns
        # -------------------------

        # Normalize overall positional columns
        df[self.OVERALL_POS_NORM_COLS] = map_graph_scaler.transform(df[self.OVERALL_POS_NORM_COLS].rename(columns={'bomb_X': 'x', 'bomb_Y': 'y'}))

        # Normalize overall columns
        for col in self.OVERALL_NORM_COLS:
            df[col] = (df[col] - overall_min_max_values[col + "_min"]) / (overall_min_max_values[col + "_max"] - overall_min_max_values[col + "_min"])




        # -------------------------
        # Normalize player columns
        # -------------------------

        # Normalize the player columns
        for player_idx in range(0, 10):

            # Normalize positional columns:  [pos_x, pos_y], [eye_x, eye_y]
            for pos_col_pair in self.PLAYER_POS_NORM_COLS:
                updated_pos_col_pair = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in pos_col_pair]
                df[updated_pos_col_pair] = map_graph_scaler.transform(df[updated_pos_col_pair].rename(columns={'player{}_x'.format(player_idx): 'x',
                                                                                                                'player{}_y'.format(player_idx): 'y',
                                                                                                                'player{}_eyeX'.format(player_idx): 'x',
                                                                                                                'player{}_eyeY'.format(player_idx): 'y'}))

            # Normalize non-positional columns
            updated_norm_cols = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in self.PLAYER_NORM_COLS]
            for col_iter_idx, col in enumerate(updated_norm_cols):
                df[col] = (df[col] - player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"]) / (player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] - player_min_max_values[player_norm_cols[col_iter_idx] + "_min"])
        


        # Return dataframe
        return df



    # Match normalization with the map scaler model path
    def match_normalization(self, df, map_graph_scaler):
        """
        Normalize one match dataset. Parameters:
            - df: The match dataset.
            - map_graph_scaler: The map graph scaler.
        """

        # Get the min-max values of all datasets
        player_min_max_values, overall_min_max_values = self.__build_scaling_dictionary__(df)


        # -------------------------
        # Normalize overall columns
        # -------------------------

        # Normalize overall positional columns
        df[self.OVERALL_POS_NORM_COLS] = map_graph_scaler.transform(df[self.OVERALL_POS_NORM_COLS].rename(columns={'bomb_X': 'x', 'bomb_Y': 'y'}))

        # Normalize overall columns
        for col in self.OVERALL_NORM_COLS:
            df[col] = (df[col] - overall_min_max_values[col + "_min"]) / (overall_min_max_values[col + "_max"] - overall_min_max_values[col + "_min"])




        # -------------------------
        # Normalize player columns
        # -------------------------

        # Normalize the player columns
        for player_idx in range(0, 10):

            # Normalize positional columns:  [pos_x, pos_y], [eye_x, eye_y]
            for pos_col_pair in self.PLAYER_POS_NORM_COLS:
                updated_pos_col_pair = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in pos_col_pair]
                df[updated_pos_col_pair] = map_graph_scaler.transform(df[updated_pos_col_pair].rename(columns={'player{}_x'.format(player_idx): 'x',
                                                                                                                'player{}_y'.format(player_idx): 'y',
                                                                                                                'player{}_eyeX'.format(player_idx): 'x',
                                                                                                                'player{}_eyeY'.format(player_idx): 'y'}))

            # Normalize non-positional columns
            updated_norm_cols = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in self.PLAYER_NORM_COLS]
            for col_iter_idx, col in enumerate(updated_norm_cols):
                df[col] = (df[col] - player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"]) / (player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] - player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"])
        


        # Return dataframe
        return df
            


    # --------------------------------------------------------------------------------------------

    # Build the scaling dictionary for collective normalization
    def __build_scaling_dictionary__(self, dataset_folder_path: str, match_list: str):

        # Min-max value store
        player_min_max_values = {}
        overall_min_max_values = {}

        # Iterate through the datasets
        for df_idx, df_name in enumerate(match_list):
            df = pd.read_csv(dataset_folder_path + df_name)

            # Iterate through the columns
            for col in self.OVERALL_NORM_COLS:

                # If it is the first iteration, create the dataset
                if df_idx == 0:
                    overall_min_max_values[col + "_min"] = df[col].min()
                    overall_min_max_values[col + "_max"] = df[col].max()

                # Else check whether the min and max values are smaller/larger than the current ones
                # If so, update the min and max values
                else:
                    overall_min_max_values[col + "_min"] = min(overall_min_max_values[col + "_min"], df[col].min())
                    overall_min_max_values[col + "_max"] = max(overall_min_max_values[col + "_max"], df[col].max())

            # Iterate through the players
            for player_idx in range(0, 10):

                # Update the player index in the column names and than iterate through the columns
                updated_norm_cols = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in self.PLAYER_NORM_COLS]
                for col_iter_idx, col in enumerate(updated_norm_cols):

                    # If it is the first iteration, create the dataset
                    if player_idx == 0 and df_idx == 0:
                        player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"] = df[col].min()
                        player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] = df[col].max()

                    # Else check whether the min and max values are smaller/larger than the current ones
                    # If so, update the min and max values
                    else:
                        player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"] = min(player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"], df[col].min())
                        player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] = max(player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"], df[col].max())

        # Return the min-max values
        return player_min_max_values, overall_min_max_values
    

    # Build the scaling dictionary for single match normalization
    def __build_scaling_dictionary__(self, df: pd.DataFrame):

        # Min-max value store
        player_min_max_values = {}
        overall_min_max_values = {}

            # Iterate through the columns
        for col in self.OVERALL_NORM_COLS:

            overall_min_max_values[col + "_min"] = df[col].min()
            overall_min_max_values[col + "_max"] = df[col].max()

        # Iterate through the players
        for player_idx in range(0, 10):

            # Update the player index in the column names and than iterate through the columns
            updated_norm_cols = [item.replace('playerNUMBER', 'player{}'.format(player_idx)) for item in self.PLAYER_NORM_COLS]
            for col_iter_idx, col in enumerate(updated_norm_cols):

                # If it is the first iteration, create the dataset
                if player_idx == 0:
                    player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"] = df[col].min()
                    player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] = df[col].max()

                # Else check whether the min and max values are smaller/larger than the current ones
                # If so, update the min and max values
                else:
                    player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"] = min(player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_min"], df[col].min())
                    player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"] = max(player_min_max_values[self.PLAYER_NORM_COLS[col_iter_idx] + "_max"], df[col].max())



        # Return the min-max values
        return player_min_max_values, overall_min_max_values