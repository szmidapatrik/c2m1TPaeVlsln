import torch
from torch_geometric.data import HeteroData

import pandas as pd
import numpy as np

import random


class SnapshotEvents:


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass 



    # --------------------------------------------------------------------------------------------
    # REGION: Public methods
    # --------------------------------------------------------------------------------------------

    def get_round_events(self, data, temp_data, model, analyzer, round_num, shift_rate=1):

        round_data = analyzer._EXT_get_round_data(data, round_num)
        predictions = analyzer.analyze_round(temp_data, model, round_num, return_predictions=True)
        round_changes = self._get_round_data(round_data, predictions, shift_rate)

        round_changes['idx'] = round_changes.reset_index().index
        return round_changes


    # --------------------------------------------------------------------------------------------
    # REGION: Private methods
    # --------------------------------------------------------------------------------------------

    
    def _get_player_data(self, graph):

        # X, Y, Z
        X = graph['player'].x[:, 0]
        Y = graph['player'].x[:, 1]
        Z = graph['player'].x[:, 2]

        # View direction
        view_X = graph['player'].x[:, 3]
        view_Y = graph['player'].x[:, 4]

        # Velocity
        vel_X = graph['player'].x[:, 5]
        vel_Y = graph['player'].x[:, 6]
        vel_Z = graph['player'].x[:, 7]

        # Health and armor
        health = graph['player'].x[:, 8]
        armor = graph['player'].x[:, 9]

        # Active weapon magazine ammo left %
        ammo_left_percent = graph['player'].x[:, 12]

        # Flashed
        flashed = graph['player'].x[:, 16]

        # Is alive and is CT
        is_alive = graph['player'].x[:, 22]
        is_ct = graph['player'].x[:, 23]

        # Shooting
        shooting = graph['player'].x[:, 24]

        # Spotted
        spotted = graph['player'].x[:, 29]

        # Is scoped
        is_scoped = graph['player'].x[:, 30]
        zoom_lvl = graph['player'].x[:, 34]

        # Is defusing
        is_defusing = graph['player'].x[:, 31]

        # Is reloading
        is_reloading = graph['player'].x[:, 32]

        # Is in bombsight
        is_in_bombsight = graph['player'].x[:, 33]

        # Stat kills
        kills = graph['player'].x[:, 36]

        # Actual nearenst map node
        positions = []
        for node in graph['map'].x:
            if str(node[0].tolist())[:2] not in positions:
                positions.append(str(node[0].tolist())[:2])
        
        position_flags = np.zeros([10, len(positions)])

        nearest_map_node = graph[('player', 'closest_to', 'map')].edge_index[1]
        for node_id in range(len(nearest_map_node)):
            position_flags[node_id][int(str(graph['map'].x[nearest_map_node[node_id]][0].tolist())[:2]) - 10] = 1

        position_flags = torch.tensor(position_flags)

        # Inventory
        inventory = graph['player'].x[:, 53:95]

        # Active weapon
        active_weapon = graph['player'].x[:, 95:138]

        # Stack the columns together
        player_data = torch.stack([X, Y, Z, view_X, view_Y, vel_X, vel_Y, vel_Z, health, armor, ammo_left_percent, flashed, is_alive, is_ct, shooting, spotted, is_scoped, zoom_lvl, is_defusing, is_reloading, is_in_bombsight, kills], dim=1)
        player_data = torch.cat([player_data, position_flags, inventory, active_weapon], dim=1)

        # Get the column names
        column_names = [
            '_X', '_Y', '_Z', '_view_X', '_view_Y', '_vel_X', '_vel_Y', '_vel_Z', '_health', '_armor', '_ammo_left_percent', '_flashed', '_is_alive', '_is_ct', '_shooting', '_spotted', '_is_scoped', '_zoom_lvl', '_is_defusing', '_is_reloading', '_is_in_bombsight', '_kills',
            '_a', '_a_balcony', '_aps', '_arch', '_b', '_back_ally', '_banana', '_bridge', '_ct_start', '_deck', '_graveyard', '_kitchen', '_library', '_lower_mid', '_mid', '_pit', '_quad', '_ruins', '_sec_mid', '_sec_mid_balcony', '_t_aps', '_t_ramp', '_t_spawn', '_top_mid', '_under', '_upstairs',
            '_inventory_C4', '_inventory_Taser', '_inventory_USP-S', '_inventory_P2000', '_inventory_Glock-18', '_inventory_Dual Berettas', '_inventory_P250', '_inventory_Tec-9', '_inventory_CZ75 Auto', '_inventory_Five-SeveN', '_inventory_Desert Eagle', '_inventory_R8 Revolver', '_inventory_MAC-10', '_inventory_MP9', '_inventory_MP7', '_inventory_MP5-SD', '_inventory_UMP-45', '_inventory_PP-Bizon', '_inventory_P90', '_inventory_Nova', '_inventory_XM1014', '_inventory_Sawed-Off', '_inventory_MAG-7', '_inventory_M249', '_inventory_Negev', '_inventory_FAMAS', '_inventory_Galil AR', '_inventory_AK-47', '_inventory_M4A4', '_inventory_M4A1-S', '_inventory_SG 553', '_inventory_AUG', '_inventory_SSG 08', '_inventory_AWP', '_inventory_G3SG1', '_inventory_SCAR-20', '_inventory_HE Grenade', '_inventory_Flashbang', '_inventory_Smoke Grenade', '_inventory_Incendiary Grenade', '_inventory_Molotov', '_inventory_Decoy Grenade', 
            '_active_weapon_C4', '_active_weapon_Knife', '_active_weapon_Taser', '_active_weapon_USP-S', '_active_weapon_P2000', '_active_weapon_Glock-18', '_active_weapon_Dual Berettas', '_active_weapon_P250', '_active_weapon_Tec-9', '_active_weapon_CZ75 Auto', '_active_weapon_Five-SeveN', '_active_weapon_Desert Eagle', '_active_weapon_R8 Revolver', '_active_weapon_MAC-10', '_active_weapon_MP9', '_active_weapon_MP7', '_active_weapon_MP5-SD', '_active_weapon_UMP-45', '_active_weapon_PP-Bizon', '_active_weapon_P90', '_active_weapon_Nova', '_active_weapon_XM1014', '_active_weapon_Sawed-Off', '_active_weapon_MAG-7', '_active_weapon_M249', '_active_weapon_Negev', '_active_weapon_FAMAS', '_active_weapon_Galil AR', '_active_weapon_AK-47', '_active_weapon_M4A4', '_active_weapon_M4A1-S', '_active_weapon_SG 553', '_active_weapon_AUG', '_active_weapon_SSG 08', '_active_weapon_AWP', '_active_weapon_G3SG1', '_active_weapon_SCAR-20', '_active_weapon_HE Grenade', '_active_weapon_Flashbang', '_active_weapon_Smoke Grenade', '_active_weapon_Incendiary Grenade', '_active_weapon_Molotov', '_active_weapon_Decoy Grenade',     
        ]

        # Create column names for all 10 players
        player_column_names = []

        for i in range(10):
            for column in column_names:
                if i < 5:
                    player_column_names.append('CT' + str(i) + column)
                else:
                    player_column_names.append('T' + str(i) + column)

        # Flatten the player data
        player_data = player_data.flatten()

        return player_column_names, player_data

    def _get_map_data(self, graph):
        
        positions = {}
        for node in graph['map'].x:
            if str(node[0].tolist())[:2] not in positions:
                positions[str(node[0].tolist())[:2]] = [node[7].tolist(), node[8].tolist()]
            else:
                positions[str(node[0].tolist())[:2]][0] += node[7].tolist()
                positions[str(node[0].tolist())[:2]][1] += node[8].tolist()

        flattened_positions = []
        pos_names = []
        for key in positions:
            pos_names.append(key + '_burning_nodes')
            flattened_positions.append(positions[key][0])
            pos_names.append(key + '_smoked_nodes')
            flattened_positions.append(positions[key][1])

        return pos_names, flattened_positions

    def _get_universal_data(self, graph):

        # Round and remaining time
        round_num = float(graph.y['round'])
        remaining_time = float(graph.y['remaining_time'])

        # Bomb dropped
        bomb_dropped = float(graph.y['is_bomb_dropped'])

        # Bomb being planted
        bomb_being_planted = float(graph.y['is_bomb_being_planted'])

        # Bomb planted on site
        bomb_on_A = float(graph.y['is_bomb_planted_at_A_site'])
        bomb_on_B = float(graph.y['is_bomb_planted_at_B_site'])

        # Bomb mx pos
        # bomb_mx_pos1 = float(graph.y['bomb_mx_pos1'])
        # bomb_mx_pos2 = float(graph.y['bomb_mx_pos2'])
        # bomb_mx_pos3 = float(graph.y['bomb_mx_pos3'])
        # bomb_mx_pos4 = float(graph.y['bomb_mx_pos4'])
        # bomb_mx_pos5 = float(graph.y['bomb_mx_pos5'])
        # bomb_mx_pos6 = float(graph.y['bomb_mx_pos6'])
        # bomb_mx_pos7 = float(graph.y['bomb_mx_pos7'])
        # bomb_mx_pos8 = float(graph.y['bomb_mx_pos8'])
        # bomb_mx_pos9 = float(graph.y['bomb_mx_pos9'])

        universal_column_names = [
            'round', 'remaining_time', 'bomb_dropped', 'bomb_being_planted', 'bomb_on_A', 'bomb_on_B' 
            # 'bomb_mx_pos1', 'bomb_mx_pos2', 'bomb_mx_pos3', 'bomb_mx_pos4', 'bomb_mx_pos5', 'bomb_mx_pos6', 'bomb_mx_pos7', 'bomb_mx_pos8', 'bomb_mx_pos9'
        ]

        universal_data = [round_num, remaining_time, bomb_dropped, bomb_being_planted, bomb_on_A, bomb_on_B] #bomb_mx_pos1, bomb_mx_pos2, bomb_mx_pos3, bomb_mx_pos4, bomb_mx_pos5, bomb_mx_pos6, bomb_mx_pos7, bomb_mx_pos8, bomb_mx_pos9]

        return universal_column_names, universal_data



    def _get_round_data(self, round_graphs, predictions, shift_rate):

        graph_infos = []
        column_names = []

        for graph in round_graphs:
            
            pn, p = self._get_player_data(graph)
            mn, m = self._get_map_data(graph)
            un, u = self._get_universal_data(graph)

            column_names = pn + mn + un
            graph_concat_data = np.concatenate([p, m, u])

            graph_infos.append(graph_concat_data)

        rdf = pd.DataFrame(graph_infos, columns=column_names)
        filler_zeros_num = len(rdf) - len(predictions)
        rdf['y'] = np.zeros(filler_zeros_num).tolist() + predictions
        round_backup = rdf['round'].copy()
        y_backup = rdf['y'].copy()

        cdf = rdf.diff().shift(-shift_rate).add_suffix('_change')

        cdf['round'] = round_backup[filler_zeros_num:-shift_rate]
        cdf['y'] = y_backup[filler_zeros_num:-shift_rate]
        cdf = cdf[filler_zeros_num:-shift_rate]

        return cdf

