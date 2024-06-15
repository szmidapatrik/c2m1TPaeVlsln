import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from IPython.display import clear_output



class TabularGraphDataNormalization:



    # Variables
    MATCH_LIST = []
    TABULAR_MATCHES_PATH = None
    OUTPUT_PATH = None

    # Column groups
    # Player positional columns, which are normalized with the min-max scaler of the inferno graph nodes
    PLAYER_POS_NORM_COLS = [['playerNUMBER_x', 'playerNUMBER_y'], ['playerNUMBER_eyeX', 'playerNUMBER_eyeY']]

    # Other player columns to normalize
    PLAYER_NORM_COLS = ['playerNUMBER_z', 'playerNUMBER_eyeZ', 'playerNUMBER_velocityX', 'playerNUMBER_velocityY', 'playerNUMBER_velocityZ',
                        'playerNUMBER_hp', 'playerNUMBER_armor',
                        'playerNUMBER_flashGrenades', 'playerNUMBER_smokeGrenades', 'playerNUMBER_heGrenades', 'playerNUMBER_totalUtility',
                        'playerNUMBER_equipmentValue', 'playerNUMBER_equipmentValueRoundStart',
                        'playerNUMBER_stat_kills', 'playerNUMBER_stat_HSK', 'playerNUMBER_stat_openKills', 'playerNUMBER_stat_tradeKills', 'playerNUMBER_stat_deaths', 'playerNUMBER_stat_openDeaths',
                        'playerNUMBER_stat_assists', 'playerNUMBER_stat_flashAssists', 'playerNUMBER_stat_damage', 'playerNUMBER_stat_weaponDamage', 'playerNUMBER_stat_nadeDamage']

    # Overall positional columns, which are normalized with the min-max scaler of the inferno graph nodes
    OVERALL_POS_NORM_COLS = ['bomb_X', 'bomb_Y']

    # Overall columns to normalize
    OVERALL_NORM_COLS = ['roundNum', 'tScore', 'ctScore', 'endTScore', 'endCTScore',
        'CT_aliveNum', 'T_aliveNum', 'CT_equipmentValue', 'T_equipmentValue',
        'bomb_Z', 'time_remaining', 'CT_totalHP', 'T_totalHP']



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    # Normalize the Inferno graph nodes dataset
    def normalize_map_graph_node_dataset(self, graph_nodes_folder_path, nodes_file_name, output_file_name=None, scaler_operation='save', model_folder_path=None, model_file_name=None):
        """
        Normalize the map graph dataset X and Y coordinates. Parameters:
            - graph_nodes_folder_path: The folder path of the graph nodes dataset.
            - nodes_file_name: The name of the graph nodes dataset.
            - output_file_name: The name of the output file.
            - scaler_operation: The operation to perform with the scaler. It can be 'save' or 'return'.
            - model_folder_path: The folder path to save the model.
            - model_file_name: The name of the model file.
        """

        # If the output file name is not provided, use the default one
        if output_file_name == None:
            output_file_name = 'norm_' + nodes_file_name

        # If the scaler operation is save, check whether the model folder path and model file name are provided
        if scaler_operation == 'save':
            if model_folder_path == None or model_file_name == None:
                raise Exception('Please provide a model folder path and a model file name.')

        nodes = pd.read_csv(graph_nodes_folder_path + nodes_file_name)
        map_graph_scaler = MinMaxScaler()

        # Fit the scaler for later use
        map_graph_scaler.fit(nodes[['x', 'y']])
        
        # Save the normalized nodes
        nodes[['x', 'y']] = map_graph_scaler.transform(nodes[['x', 'y']])
        nodes.to_csv(graph_nodes_folder_path + output_file_name, index=False)
        
        if scaler_operation == 'save':
            # Save the scaler for later use
            dump(map_graph_scaler, model_folder_path + model_file_name)

        elif scaler_operation == 'return':
            return map_graph_scaler
    


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
    def match_normalization(self, match_dataset, map_graph_scaler_path: str):
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
        map_graph_scaler = load(map_graph_scaler_path)
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