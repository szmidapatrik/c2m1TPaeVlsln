import torch
from torch_geometric.data import HeteroData

import pandas as pd
import numpy as np

import random


class CS2_GraphSnapshots:



    # ----------------------------------------------

    def __init__(self):
        pass 



    # ----------------------------------------------

    def process_snapshots(self, df: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, player_edges_num: int = 1):
        """
        Create graphs from the rows of a tabular snapshot dataframe.
        
        Parameters:
        - df: the snapshot dataframe.
        - nodes: the map graph nodes dataframe.
        - edges: the map graph edges dataframe.
        - player_edges_num: the number of closest nodes the player should be connected to in the graph. Default is 1.
        """

        # --------------------------------------------------
        #      0. Validation, create needed variables
        # --------------------------------------------------

        # Validate the input paramters
        self._PREP_validate_inputs_(df, nodes, edges, player_edges_num)

        # Create a list to store the heterogeneous graph snapshots
        heterograph_snapshot_list = []

        # Store the actual round number and the last round number the 'bomb near' value was calculated for
        actual_round_num = 0
        last_round_bomb_near_was_calculated_for = 0

        # Columns of the nodes dataframe
        nodes_columns = ['X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning']

        # Node dataframe to use for the graph
        nodes_for_graph = nodes.copy() 



        # -------------------- ITERATION --------------------
        # Iterate through the rows of the snapshot dataframe
        for row_idx in range(0, len(df)):

            # ROW
            row = df.iloc[row_idx]



            # --------------------------------------------------
            #         1. Create the accurate node dataset
            # --------------------------------------------------

            # Set actual round number
            actual_round_num = row['UNIVERSAL_round']

            # If the bomb isn't planted, use the original nodes
            if row['UNIVERSAL_is_bomb_planted_at_A_site'] + row['UNIVERSAL_is_bomb_planted_at_B_site'] == 0:
                nodes_for_graph = nodes.copy()
                
            # If the bomb is planted, and the nodes near the bomb weren't calculated yet for this round, calculate them
            elif (row['UNIVERSAL_is_bomb_planted_at_A_site'] + row['UNIVERSAL_is_bomb_planted_at_B_site'] == 1) and (actual_round_num != last_round_bomb_near_was_calculated_for):
                nodes_for_graph = self._EXT_set_bomb_planted_near_for_nodes_(nodes.copy(), df.copy(), row_idx)
                last_round_bomb_near_was_calculated_for = actual_round_num
            # else:
                # If the bomb is planted, and the nodes near the bomb were calculated already for this round, use the nodes_for_graph dataframe



            # --------------------------------------------------
            #       2. Get player nodes and edges tensors
            # --------------------------------------------------

            # Get the tensors for the graph
            player_tensor = self._PLAYER_nodes_tensor_(row)
            player_edges_tensor = self._PLAYER_edges_tensor_(row, nodes_for_graph)



            # --------------------------------------------------
            #         3. Create the heterodata object
            # --------------------------------------------------

            # Create a HeteroData object
            data = HeteroData()

            # Create node data
            data['player'].x = torch.tensor(player_tensor, dtype=torch.float32)
            data['map'].x = torch.tensor(nodes_for_graph[nodes_columns].values, dtype=torch.float32)

            # Create edge data
            data['map', 'connected_to', 'map'].edge_index = torch.tensor(edges.values.T, dtype=torch.int16)
            data['player', 'closest_to', 'map'].edge_index = torch.tensor(player_edges_tensor, dtype=torch.int16)

            # Define the graph-level features
            data.y = {
                'numerical_match_id': row['NUMERICAL_MATCH_ID'].astype('float32'),
                'tick': row['UNIVERSAL_tick'].astype('float32'),
                'round': row['UNIVERSAL_round'].astype('float32'),
                'time': row['UNIVERSAL_time'].astype('float32'),
                'remaining_time': row['UNIVERSAL_remaining_time'].astype('float32'),
                'freeze_end': row['UNIVERSAL_freeze_end'].astype('float32'),
                'end': row['UNIVERSAL_end'].astype('float32'),
                'CT_alive_num': row['UNIVERSAL_CT_alive_num'].astype('float32'),
                'T_alive_num': row['UNIVERSAL_T_alive_num'].astype('float32'),
                'CT_total_hp': row['UNIVERSAL_CT_total_hp'].astype('float32'),
                'T_total_hp': row['UNIVERSAL_T_total_hp'].astype('float32'),
                'CT_equipment_value': row['UNIVERSAL_CT_equipment_value'].astype('float32'),
                'T_equipment_value': row['UNIVERSAL_T_equipment_value'].astype('float32'),
                'CT_losing_streak': row['UNIVERSAL_CT_losing_streak'].astype('float32'),
                'T_losing_streak': row['UNIVERSAL_T_losing_streak'].astype('float32'),
                'is_bomb_dropped': row['UNIVERSAL_is_bomb_dropped'].astype('float16'),
                'is_bomb_being_planted': row['UNIVERSAL_is_bomb_being_planted'].astype('float16'),
                'is_bomb_being_defused': row['UNIVERSAL_is_bomb_being_defused'].astype('float16'),
                'is_bomb_defused': row['UNIVERSAL_is_bomb_defused'].astype('float16'),
                'is_bomb_planted_at_A_site': row['UNIVERSAL_is_bomb_planted_at_A_site'].astype('float16'),
                'is_bomb_planted_at_B_site': row['UNIVERSAL_is_bomb_planted_at_B_site'].astype('float16'),
                'bomb_X': row['UNIVERSAL_bomb_X'].astype('float32'),
                'bomb_Y': row['UNIVERSAL_bomb_Y'].astype('float32'),
                'bomb_Z': row['UNIVERSAL_bomb_Z'].astype('float32'),
                'bomb_mx_pos1': row['UNIVERSAL_bomb_mx_pos1'].astype('float16'),
                'bomb_mx_pos2': row['UNIVERSAL_bomb_mx_pos2'].astype('float16'),
                'bomb_mx_pos3': row['UNIVERSAL_bomb_mx_pos3'].astype('float16'),
                'bomb_mx_pos4': row['UNIVERSAL_bomb_mx_pos4'].astype('float16'),
                'bomb_mx_pos5': row['UNIVERSAL_bomb_mx_pos5'].astype('float16'),
                'bomb_mx_pos6': row['UNIVERSAL_bomb_mx_pos6'].astype('float16'),
                'bomb_mx_pos7': row['UNIVERSAL_bomb_mx_pos7'].astype('float16'),
                'bomb_mx_pos8': row['UNIVERSAL_bomb_mx_pos8'].astype('float16'),
                'bomb_mx_pos9': row['UNIVERSAL_bomb_mx_pos9'].astype('float16'),
                'CT_wins': row['UNIVERSAL_CT_wins'].astype('float16'),
            }



            # Append the HeteroData object to the list
            heterograph_snapshot_list.append(data)

            # Clear the memory
            del data
            del player_tensor
            del player_edges_tensor



        # Return the list of HeteroData objects
        return heterograph_snapshot_list

    # ----------------------------------------------

    # 0. Validate the input parameters
    def _PREP_validate_inputs_(self, df: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, player_edges_num: int):

        # Check if the input parameters are empty
        if df.empty:
            raise ValueError("The snapshot dataframe is empty.")
        if nodes.empty:
            raise ValueError("The nodes dataframe is empty.")
        if edges.empty:
            raise ValueError("The edges dataframe is empty.")
        
        # Check if the nodes dataset containes the required columns
        if not all(col in nodes.columns for col in ['node_id', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning']):
            raise ValueError("The nodes dataframe does not contain the required columns. Required columns are: 'node_id', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning'.")
        
        # Check if the edges dataset containes the required columns
        if not all(col in edges.columns for col in ['source', 'target']):
            raise ValueError("The edges dataframe does not contain the required columns. Required columns are: 'source', 'target'.")
        
        # Check if the player_edges_num is a positive integer
        if not isinstance(player_edges_num, int) or player_edges_num < 1:
            raise ValueError("The player_edges_num should be a positive integer.")

    # 1. Set the 'is_bomb_planted_near' value for the nodes near the bomb
    def _EXT_set_bomb_planted_near_for_nodes_(self, nodes, df, index):
        closest_node_id = self.__EXT_closest_node_to_pos__(df.iloc[index]['UNIVERSAL_bomb_X'], df.iloc[index]['UNIVERSAL_bomb_Y'], df.iloc[index]['UNIVERSAL_bomb_Z'], nodes)
        nodes.loc[nodes['node_id'] == closest_node_id, 'UNIVERSAL_is_bomb_planted_near'] = 1
        return nodes

    # 2.1 Create the player nodes tensor
    def _PLAYER_nodes_tensor_(self, row):

        # Drop the player name columns
        drop_cols = [
            'CT0_name', 'CT1_name', 'CT2_name', 'CT3_name', 'CT4_name',
            'T5_name', 'T6_name', 'T7_name', 'T8_name', 'T9_name',
        ]
        row = row.drop(labels=drop_cols)

        # Create the players tensor
        players_tensor = np.array([])

        # Iterate through the players
        for i in range(0, 10):
            
            # Get the columns for the player with the right index
            if i < 5:
                player_columns = [col for col in row.keys() if f'CT{i}' in col]
            else:
                player_columns = [col for col in row.keys() if f'T{i}' in col]

            # Get the player data
            player = row[player_columns].values

            # Add the player data to the players tensor
            if len(players_tensor) == 0:
                players_tensor = np.array([player])
            else:
                players_tensor = np.vstack([players_tensor, player])
            
        return players_tensor

    # 2.2 Create the player edges tensor
    def _PLAYER_edges_tensor_(self, row, nodes):
    
        # Get the nearest node ids for each player
        nearest_nodes_arr = np.array([])
        for player_idx in range(0, 10):
            if (player_idx < 5):
                nearest_node = self.__EXT_closest_node_to_pos__(row[f'CT{player_idx}_X'], row[f'CT{player_idx}_Y'], row[f'CT{player_idx}_Z'], nodes)
                nearest_nodes_arr = np.append(nearest_nodes_arr, nearest_node)
            else:
                nearest_node = self.__EXT_closest_node_to_pos__(row[f'T{player_idx}_X'], row[f'T{player_idx}_Y'], row[f'T{player_idx}_Z'], nodes)
                nearest_nodes_arr = np.append(nearest_nodes_arr, nearest_node)


        playerEdges = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            nearest_nodes_arr
        ])
        return playerEdges

    

    # ----------------------------------------------
  
    # Calculate closest graph node to a position
    def __EXT_closest_node_to_pos__(self, coord_x, coord_y, coord_z, nodes):
        """
        Returns the id of the closest node to a given position.
        
        Parameters:
        - coord_x: the x coordinate of the position.
        - coord_y: the y coordinate of the position.
        - coord_z: the z coordinate of the position.
        - nodes: the nodes dataframe.
        """
        distances = np.sqrt((nodes['X'] - coord_x)**2 + (nodes['Y'] - coord_y)**2 + (nodes['Z'] - coord_z)**2)
        return nodes.loc[distances.idxmin(), 'node_id']


