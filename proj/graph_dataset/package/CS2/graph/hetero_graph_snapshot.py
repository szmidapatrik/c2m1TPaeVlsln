import torch
from torch_geometric.data import HeteroData

import pandas as pd
import numpy as np

import random


class HeteroGraphSnapshot:

    # Molotov and incendiary grenade radius values
    MOLOTOV_RADIUS_X = None
    MOLOTOV_RADIUS_Y = None
    MOLOTOV_RADIUS_Z = None
    
    # Smoke grenade radius values
    SMOKE_RADIUS_X = None
    SMOKE_RADIUS_Y = None
    SMOKE_RADIUS_Z = None


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass 



    # --------------------------------------------------------------------------------------------
    # REGION: Public methods
    # --------------------------------------------------------------------------------------------

    def process_snapshots(
        self, 
        df: pd.DataFrame, 
        nodes: pd.DataFrame, 
        edges_pos_id: pd.DataFrame, 
        active_infernos: pd.DataFrame,
        active_smokes: pd.DataFrame,
        active_he_explosions: pd.DataFrame,
        CONFIG_MOLOTOV_RADIUS: dict,
        CONFIG_SMOKE_RADIUS: dict,
        player_edges_num: int = 1
    ):
        """
        Create graphs from the rows of a tabular snapshot dataframe.
        
        Parameters:
        - df: the snapshot dataframe.
        - nodes: the map graph nodes dataframe.
        - edges: the map graph edges dataframe.
        - active_infernos: the active infernos dataframe.
        - active_smokes: the active smokes dataframe.
        - actigve_he_explosions: the active HE grenade explosions dataframe.
        - CONFIG_MOLOTOV_RADIUS: the molotov and incendiary grenade radius values.
        - CONFIG_SMOKE_RADIUS: the smoke grenade radius values.
        - player_edges_num: the number of closest nodes the player should be connected to in the graph. Default is 1.
        """



        # ---- 0. Validation, create needed variables ------

        # Validate the input paramters and create the accurate edges dataframe
        self._PREP_validate_inputs_(df, nodes, edges_pos_id, CONFIG_MOLOTOV_RADIUS, player_edges_num)
        self._PREP_set_molotov_radius_(CONFIG_MOLOTOV_RADIUS)
        self._PREP_set_smoke_radius_(CONFIG_SMOKE_RADIUS)
        edges = self._PREP_create_edges_(nodes, edges_pos_id)

        # Create a list to store the heterogeneous graph snapshots
        heterograph_snapshot_list = []

        # Store the actual round number and the last round number the 'bomb near' value was calculated for
        actual_round_num = 0
        last_round_bomb_near_was_calculated_for = 0

        # Columns of the nodes dataframe
        nodes_columns = ['pos_id', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning', 'is_smoked']

        # Node dataframe to use for the graph
        nodes_bomb = nodes.copy() 



        # -------------------- ITERATION --------------------
        # Iterate through the rows of the snapshot dataframe
        for row_idx in range(0, len(df)):

            # ROW and TICK
            row = df.iloc[row_idx]
            tick = row['UNIVERSAL_tick']

            

            # ------- 1. Create the accurate node dataset ------

            # -------- 1.1 Set the nodes near the bomb ---------

            # Set actual round number
            actual_round_num = row['UNIVERSAL_round']

            # If the bomb isn't planted, use the original nodes
            if row['UNIVERSAL_is_bomb_planted_at_A_site'] + row['UNIVERSAL_is_bomb_planted_at_B_site'] == 0:
                nodes_bomb = nodes.copy()
                
            # If the bomb is planted, and the nodes near the bomb weren't calculated yet for this round, calculate them
            elif (row['UNIVERSAL_is_bomb_planted_at_A_site'] + row['UNIVERSAL_is_bomb_planted_at_B_site'] == 1) and (actual_round_num != last_round_bomb_near_was_calculated_for):
                nodes_bomb = self._EXT_set_bomb_planted_near_for_nodes_(nodes.copy(), df.copy(), row_idx)
                last_round_bomb_near_was_calculated_for = actual_round_num
            # else:
                # If the bomb is planted, and the nodes near the bomb were calculated already for this round, use the nodes_with_bomb dataframe



            # ------- 1.2 Set the nodes that are burning -------

            nodes_with_bomb_inf = self._EXT_set_burning_(nodes_bomb, active_infernos, tick)



            # -------- 1.3 Add the smokes to the map -----------

            nodes_with_bomb_inf_smokes = self._EXT_set_smokes_(nodes_with_bomb_inf, active_smokes, active_he_explosions, tick)



            # ---- 2. Get player nodes and edges tensors -------

            # Get the tensors for the graph
            player_tensor = self._PLAYER_nodes_tensor_(row)
            player_edges_tensor = self._PLAYER_edges_tensor_(row, nodes_with_bomb_inf_smokes)



            # ------- 3. Create the heterodata object ----------

            # Create a HeteroData object
            data = HeteroData()

            # Create node data
            data['player'].x = torch.tensor(player_tensor, dtype=torch.float32)
            data['map'].x = torch.tensor(nodes_with_bomb_inf_smokes[nodes_columns].values, dtype=torch.float32)

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



    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------

    # 0. Validate the input parameters
    def _PREP_validate_inputs_(self, df: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, CONFIG_MOLOTOV_RADIUS: dict, player_edges_num: int):

        # Check if the input parameters are empty
        if df.empty:
            raise ValueError("The snapshot dataframe is empty.")
        if nodes.empty:
            raise ValueError("The nodes dataframe is empty.")
        if edges.empty:
            raise ValueError("The edges dataframe is empty.")
        
        # Check if the nodes dataset containes the required columns
        if not all(col in nodes.columns for col in ['pos_id', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning']):
            raise ValueError("The nodes dataframe does not contain the required columns. Required columns are: 'node_id', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning'.")
        
        # Check if the edges dataset containes the required columns
        if not all(col in edges.columns for col in ['source_pos_id', 'source_pos_id']):
            raise ValueError("The edges dataframe does not contain the required columns. Required columns are: 'source', 'target'.")
        
        # Check if the CONFIG_MOLOTOV_RADIUS is a dictionary
        if not isinstance(CONFIG_MOLOTOV_RADIUS, dict):
            raise ValueError("The CONFIG_MOLOTOV_RADIUS should be a dictionary.")
        
        # Check if the CONFIG_MOLOTOV_RADIUS contains the required keys
        if not all(key in CONFIG_MOLOTOV_RADIUS for key in ['X', 'Y', 'Z']):
            raise ValueError("The CONFIG_MOLOTOV_RADIUS dictionary should contain the keys 'X', 'Y', 'Z'.")

        # Check if the player_edges_num is a positive integer
        if not isinstance(player_edges_num, int) or player_edges_num < 1:
            raise ValueError("The player_edges_num should be a positive integer.")

        # 0.2 Set the molotov and incendiary grenade radius values
    
    # 0.1 Set the molotov and incendiary grenade radius values
    def _PREP_set_molotov_radius_(self, CONFIG_MOLOTOV_RADIUS: dict):
        
        self.MOLOTOV_RADIUS_X = CONFIG_MOLOTOV_RADIUS['X']
        self.MOLOTOV_RADIUS_Y = CONFIG_MOLOTOV_RADIUS['Y']
        self.MOLOTOV_RADIUS_Z = CONFIG_MOLOTOV_RADIUS['Z']
    
    # 0.2 Set the smoke grenade radius values
    def _PREP_set_smoke_radius_(self, CONFIG_SMOKE_RADIUS: dict):
        
        self.SMOKE_RADIUS_X = CONFIG_SMOKE_RADIUS['X']
        self.SMOKE_RADIUS_Y = CONFIG_SMOKE_RADIUS['Y']
        self.SMOKE_RADIUS_Z = CONFIG_SMOKE_RADIUS['Z']

    # 0.2 Create the edges dataframe
    def _PREP_create_edges_(self, nodes: pd.DataFrame, edges_pos_id: pd.DataFrame):

        # Add node_id column to the nodes dataframe by setting the index as the node_id, create a copy of the edges dataframe
        nodes['node_id'] = nodes.index
        edges = edges_pos_id.copy()

        # Merge the nodes dataframe with the edges dataframe to get the source node ids
        edges = edges.merge(nodes[['pos_id', 'node_id']], left_on='source_pos_id', right_on='pos_id', how='left')
        edges = edges.drop(columns=['pos_id'])
        edges = edges.rename(columns={'node_id': 'source'})

        # Merge the nodes dataframe with the edges dataframe to get the target node ids
        edges = edges.merge(nodes[['pos_id', 'node_id']], left_on='target_pos_id', right_on='pos_id', how='left')
        edges = edges.drop(columns=['pos_id'])
        edges = edges.rename(columns={'node_id': 'target'})

        # Delete the source_pos_id and target_pos_id columns
        del edges['source_pos_id']
        del edges['target_pos_id']

        return edges

    # 1.1 Set the 'is_bomb_planted_near' value for the nodes near the bomb
    def _EXT_set_bomb_planted_near_for_nodes_(self, nodes, df, index):
        closest_node_id = self.__EXT_closest_node_to_pos__(df.iloc[index]['UNIVERSAL_bomb_X'], df.iloc[index]['UNIVERSAL_bomb_Y'], df.iloc[index]['UNIVERSAL_bomb_Z'], nodes)
        nodes.loc[nodes['pos_id'] == closest_node_id, 'UNIVERSAL_is_bomb_planted_near'] = 1
        return nodes

    # 1.2 Set the 'is_burning' value for the nodes that are burning
    def _EXT_set_burning_(self, nodes_bomb: pd.DataFrame, active_infernos: pd.DataFrame, tick: int):

        # Reset the 'is_burning' value for all nodes
        nodes_bomb['is_burning'] = 0

        # Select the actual rows from the infernos dataframe
        active_infernos = active_infernos[active_infernos['tick'] == tick]

        # If there are no molotovs thrown, continue
        if len(active_infernos) == 0:
            return nodes_bomb

        # If there are molotovs thrown, set the 'is_burning' values
        else:
            # Iterate through the molotovs
            for _, molotov in active_infernos.iterrows():

                # Nodes close to the molotov
                nodes_close = nodes_bomb.loc[
                    (nodes_bomb['X'] >= (molotov['X'] - self.MOLOTOV_RADIUS_X)) & (nodes_bomb['X'] <= (molotov['X'] + self.MOLOTOV_RADIUS_X)) &
                    (nodes_bomb['Y'] >= (molotov['Y'] - self.MOLOTOV_RADIUS_Y)) & (nodes_bomb['Y'] <= (molotov['Y'] + self.MOLOTOV_RADIUS_Y)) &
                    (nodes_bomb['Z'] >= (molotov['Z'] - self.MOLOTOV_RADIUS_Z)) & (nodes_bomb['Z'] <= (molotov['Z'] + self.MOLOTOV_RADIUS_Z))]
                
                # If there are no nodes close to the molotov, get the closest node
                if len(nodes_close) == 0:
                    closest_node_id = self.__EXT_closest_node_to_pos__(molotov['X'], molotov['Y'], molotov['Z'], nodes_bomb)
                    nodes_bomb.loc[nodes_bomb['pos_id'] == closest_node_id, 'is_burning'] = 1
                
                # If there are nodes close to the molotov, set the 'is_burning' value for them
                else:
                    nodes_bomb.loc[
                        (nodes_bomb['X'] >= (molotov['X'] - self.MOLOTOV_RADIUS_X)) & (nodes_bomb['X'] <= (molotov['X'] + self.MOLOTOV_RADIUS_X)) &
                        (nodes_bomb['Y'] >= (molotov['Y'] - self.MOLOTOV_RADIUS_Y)) & (nodes_bomb['Y'] <= (molotov['Y'] + self.MOLOTOV_RADIUS_Y)) &
                        (nodes_bomb['Z'] >= (molotov['Z'] - self.MOLOTOV_RADIUS_Z)) & (nodes_bomb['Z'] <= (molotov['Z'] + self.MOLOTOV_RADIUS_Z)),
                        'is_burning'] = 1
            
        return nodes_bomb

    # 1.3 Add smokes to the graph
    def _EXT_set_smokes_(self, nodes_bomb_inf: pd.DataFrame, active_smokes: pd.DataFrame, active_he_explosions: pd.DataFrame, tick: int):
        
        # Reset the 'is_smoked' value for all nodes
        nodes_bomb_inf['is_smoked'] = 0

        # Select the actual rows from the smokes dataframe
        active_smokes = active_smokes[active_smokes['tick'] == tick]

        # If there are no smokes thrown, continue
        if len(active_smokes) == 0:
            return nodes_bomb_inf

        # If there are smokes thrown, check whether any HE grenade explosions are near for each smoke
        else:
            # Iterate through the smokes
            for _, smoke in active_smokes.iterrows():

                # Skip flag for the actual smoke if there is an HE grenade explosion near
                SKIP_SMOKE = False

                # Select the actual rows from the HE dataframe
                active_he_explosions = active_he_explosions[active_he_explosions['tick'] == tick]

                # If there are nade explosions in the actual tick, check whether any of them are near the smoke
                if len(active_he_explosions) != 0:
                    
                    # Iterate through the HE grenade explosions
                    for _, he_explosion in active_he_explosions.iterrows():

                        # Check if the HE grenade explosion is near the smoke
                        if  (he_explosion['X'] >= (smoke['X'] - self.SMOKE_RADIUS_X)) & (he_explosion['X'] <= (smoke['X'] + self.SMOKE_RADIUS_X)) & \
                            (he_explosion['Y'] >= (smoke['Y'] - self.SMOKE_RADIUS_Y)) & (he_explosion['Y'] <= (smoke['Y'] + self.SMOKE_RADIUS_Y)) & \
                            (he_explosion['Z'] >= (smoke['Z'] - self.SMOKE_RADIUS_Z)) & (he_explosion['Z'] <= (smoke['Z'] + self.SMOKE_RADIUS_Z)):
                            
                            # Set the skip flag if the HE grenade explosion is near the smoke
                            SKIP_SMOKE = True

                # If there are HE explosions in the smoke radius, skip the smoke, else set the 'is_smoked' value for the nodes
                if SKIP_SMOKE:
                    continue

                else:

                    # Get the nodes close to the smoke
                    nodes_close_to_smoke = nodes_bomb_inf.loc[
                        (nodes_bomb_inf['X'] >= (smoke['X'] - self.SMOKE_RADIUS_X)) & (nodes_bomb_inf['X'] <= (smoke['X'] + self.SMOKE_RADIUS_X)) &
                        (nodes_bomb_inf['Y'] >= (smoke['Y'] - self.SMOKE_RADIUS_Y)) & (nodes_bomb_inf['Y'] <= (smoke['Y'] + self.SMOKE_RADIUS_Y)) &
                        (nodes_bomb_inf['Z'] >= (smoke['Z'] - self.SMOKE_RADIUS_Z)) & (nodes_bomb_inf['Z'] <= (smoke['Z'] + self.SMOKE_RADIUS_Z))]
                    
                    # If there are no nodes close to the smoke, get the closest node
                    if len(nodes_close_to_smoke) == 0:
                        closest_node_id = self.__EXT_closest_node_to_pos__(smoke['X'], smoke['Y'], smoke['Z'], nodes_bomb_inf)
                        nodes_bomb_inf.loc[nodes_bomb_inf['pos_id'] == closest_node_id, 'is_smoked'] = 1

                    else:
                        nodes_bomb_inf.loc[
                            (nodes_bomb_inf['X'] >= (smoke['X'] - self.SMOKE_RADIUS_X)) & (nodes_bomb_inf['X'] <= (smoke['X'] + self.SMOKE_RADIUS_X)) &
                            (nodes_bomb_inf['Y'] >= (smoke['Y'] - self.SMOKE_RADIUS_Y)) & (nodes_bomb_inf['Y'] <= (smoke['Y'] + self.SMOKE_RADIUS_Y)) &
                            (nodes_bomb_inf['Z'] >= (smoke['Z'] - self.SMOKE_RADIUS_Z)) & (nodes_bomb_inf['Z'] <= (smoke['Z'] + self.SMOKE_RADIUS_Z)),
                            'is_smoked'] = 1
            
        return nodes_bomb_inf




    # 2.1 Create the player nodes tensor
    def _PLAYER_nodes_tensor_(self, row):

        # Drop the player name columns
        drop_cols = [
            'CT0_name', 'CT1_name', 'CT2_name', 'CT3_name', 'CT4_name',
            'T5_name', 'T6_name', 'T7_name', 'T8_name', 'T9_name',
        ]
        row = row.drop(labels=drop_cols)

        # Create the players tensor
        players_tensor = np.array([], dtype=np.float32)

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
            
        return players_tensor.astype(np.float32)

    # 2.2 Create the player edges tensor
    def _PLAYER_edges_tensor_(self, row, nodes):
    
        # Get the nearest node ids for each player
        nearest_nodes_arr = np.array([])
        for player_idx in range(0, 10):
            if (player_idx < 5):
                nearest_node = self.__EXT_closest_node_to_player__(row[f'CT{player_idx}_X'], row[f'CT{player_idx}_Y'], row[f'CT{player_idx}_Z'], nodes)
                nearest_nodes_arr = np.append(nearest_nodes_arr, nearest_node)
            else:
                nearest_node = self.__EXT_closest_node_to_player__(row[f'T{player_idx}_X'], row[f'T{player_idx}_Y'], row[f'T{player_idx}_Z'], nodes)
                nearest_nodes_arr = np.append(nearest_nodes_arr, nearest_node)


        playerEdges = np.array([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            nearest_nodes_arr
        ])
        return playerEdges

    

    # --------------------------------------------------------------------------------------------
    # REGION: External methods
    # --------------------------------------------------------------------------------------------
  
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
        return nodes.loc[distances.idxmin(), 'pos_id']
    
    # Calculate closest graph node to a position
    def __EXT_closest_node_to_player__(self, coord_x, coord_y, coord_z, nodes):
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


