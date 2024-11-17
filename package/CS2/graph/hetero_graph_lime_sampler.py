import torch
from torch_geometric.data import HeteroData

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import pandas as pd
import numpy as np

import random
import os

class HeteroGraphLIMESampler:


    player_columns = [
        'X', 'Y', 'Z', 'pitch', 'yaw', 'velocity_X', 'velocity_Y', 'velocity_Z', 'health', 'armor_value', 'active_weapon_magazine_size', 'active_weapon_ammo', 'active_weapon_magazine_ammo_left_%', 'active_weapon_max_ammo', 'total_ammo_left', 'active_weapon_total_ammo_left_%', 'flash_duration', 'flash_max_alpha', 'balance', 'current_equip_value', 'round_start_equip_value', 'cash_spent_this_round', 'is_alive',
        'is_CT', 'is_shooting', 'is_crouching', 'is_ducking', 'is_duck_jumping', 'is_walking', 'is_spotted', 'is_scoped', 'is_defusing', 'is_reloading', 'is_in_bombsite', 
        'zoom_lvl', 'velo_modifier',
        'stat_kills', 'stat_HS_kills', 'stat_opening_kills', 'stat_MVPs', 'stat_deaths', 'stat_opening_deaths', 'stat_assists', 'stat_flash_assists', 'stat_damage', 'stat_weapon_damage', 'stat_nade_damage', 'stat_survives', 'stat_KPR', 'stat_ADR', 'stat_DPR', 'stat_HS%', 'stat_SPR', 'inventory_C4',
        'inventory_Taser', 'inventory_USP-S', 'inventory_P2000', 'inventory_Glock-18', 'inventory_Dual Berettas', 'inventory_P250', 'inventory_Tec-9', 'inventory_CZ75 Auto', 'inventory_Five-SeveN', 'inventory_Desert Eagle', 'inventory_R8 Revolver', 'inventory_MAC-10', 'inventory_MP9', 'inventory_MP7', 'inventory_MP5-SD', 'inventory_UMP-45', 'inventory_PP-Bizon', 'inventory_P90', 'inventory_Nova', 'inventory_XM1014', 'inventory_Sawed-Off', 'inventory_MAG-7', 'inventory_M249', 'inventory_Negev', 'inventory_FAMAS', 'inventory_Galil AR', 'inventory_AK-47', 'inventory_M4A4', 'inventory_M4A1-S', 'inventory_SG 553', 'inventory_AUG', 'inventory_SSG 08', 'inventory_AWP', 'inventory_G3SG1', 'inventory_SCAR-20', 'inventory_HE Grenade', 'inventory_Flashbang', 'inventory_Smoke Grenade', 'inventory_Incendiary Grenade', 'inventory_Molotov', 'inventory_Decoy Grenade', 'active_weapon_C4',
        'active_weapon_Knife', 'active_weapon_Taser', 'active_weapon_USP-S', 'active_weapon_P2000', 'active_weapon_Glock-18', 'active_weapon_Dual Berettas', 'active_weapon_P250', 'active_weapon_Tec-9', 'active_weapon_CZ75 Auto', 'active_weapon_Five-SeveN', 'active_weapon_Desert Eagle', 'active_weapon_R8 Revolver', 'active_weapon_MAC-10', 'active_weapon_MP9', 'active_weapon_MP7', 'active_weapon_MP5-SD', 'active_weapon_UMP-45', 'active_weapon_PP-Bizon', 'active_weapon_P90', 'active_weapon_Nova', 'active_weapon_XM1014', 'active_weapon_Sawed-Off', 'active_weapon_MAG-7', 'active_weapon_M249', 'active_weapon_Negev', 'active_weapon_FAMAS', 'active_weapon_Galil AR', 'active_weapon_AK-47', 'active_weapon_M4A4', 'active_weapon_M4A1-S', 'active_weapon_SG 553', 'active_weapon_AUG', 'active_weapon_SSG 08', 'active_weapon_AWP', 'active_weapon_G3SG1', 'active_weapon_SCAR-20', 'active_weapon_HE Grenade', 'active_weapon_Flashbang', 'active_weapon_Smoke Grenade', 'active_weapon_Incendiary Grenade', 'active_weapon_Molotov', 'active_weapon_Decoy Grenade', 'hltv_rating_2.0',
        'hltv_DPR', 'hltv_KAST', 'hltv_Impact', 'hltv_ADR', 'hltv_KPR', 'hltv_total_kills', 'hltv_HS%', 'hltv_total_deaths', 'hltv_KD_ratio', 'hltv_dmgPR', 'hltv_grenade_dmgPR', 'hltv_maps_played', 'hltv_saved_by_teammatePR', 'hltv_saved_teammatesPR', 'hltv_opening_kill_rating', 'hltv_team_W%_after_opening', 'hltv_opening_kill_in_W_rounds', 'hltv_rating_1.0_all_Career', 'hltv_clutches_1on1_ratio', 'hltv_clutches_won_1on1', 'hltv_clutches_won_1on2', 'hltv_clutches_won_1on3', 'hltv_clutches_won_1on4', 'hltv_clutches_won_1on5'
    ]


    # --------------------------------------------------------------------------------------------
    # REGION: Constructor
    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------
    # REGION: Public functions - Sample
    # --------------------------------------------------------------------------------------------

    # Visualize a heterogeneous graph snapshot
    def sample_snapshot(self, graph: HeteroData, sample_size: int, probability: float = 0.1, normalized: bool = True):
        """
        Create a LIME sampling for a heterogeneous graph snapshot.
        Parameters:
        - graph: the HeteroData graph to sample.
        - sample_size: the number of samples to generate.
        - map: the map on which the match was held.
        - normalized: whether the input graph is normalized. Default is True.
        """


        # -------------------------------------------------
        # Validate inputs
        # ------------------------------------------------

        self.validate_inputs(graph, sample_size, normalized)

        
        # -------------------------------------------------
        # Create the similar game-state samples
        # -------------------------------------------------

        samples = []
        for _ in range(sample_size):
            samples.append(graph.clone())

        samples = self._update_player_tensor(samples, probability)
        samples = self._update_player_map_edges(samples)
        samples = self._update_map_node_burning_smoked_values(samples)
        samples = self._update_y_values(samples)

        return samples



    # --------------------------------------------------------------------------------------------
    # REGION: Private functions
    # --------------------------------------------------------------------------------------------

    # Validate the inputs
    def validate_inputs(self, graph: HeteroData, sample_size: int, normalized: bool):

        # Validate sample size
        if sample_size <= 0:
            raise ValueError('Invalid sample size. Must be a positive integer.')

        # Validate graph
        if not isinstance(graph, HeteroData):
            raise ValueError('Invalid graph. Must be a HeteroData object.')

        # Validate normalized
        if not isinstance(normalized, bool):
            raise ValueError('Invalid normalized. Must be a boolean.')
        
        # Normalized must be true
        if not normalized:
            raise ValueError('Normalized must be True. Only normalized graphs are supported at this point.')



    # Update the player tensor of the graph
    def _update_player_tensor(self, samples: list, probability: float):
        """
        Update the player tensor of the graph.
        Parameters:
        - samples: the list of HeteroData samples to update.
        - map_name: the name of the map.
        - normalized: whether the input graph is normalized.
        """

        # Update the player tensor for each sample
        for sample in samples:
            
            tpldf = pd.DataFrame(sample.x_dict['player'].numpy(), columns=self.player_columns)

            # ----------- Player coordinates ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['X'] = np.where(random_filter, 
                                    tpldf['X'] + np.random.normal(0, 0.006, tpldf.shape[0]), 
                                    tpldf['X'])
            tpldf['Y'] = np.where(random_filter, 
                                    tpldf['Y'] + np.random.normal(0, 0.006, tpldf.shape[0]), 
                                    tpldf['Y'])



            # ----------- Player view directions ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['pitch'] = np.where(random_filter, 
                                        (tpldf['pitch'] + np.random.normal(0, 0.2, tpldf.shape[0])).clip(0, 1), 
                                        tpldf['pitch'])
            tpldf['yaw'] = np.where(random_filter, 
                                        (tpldf['yaw'] + np.random.normal(0, 0.2, tpldf.shape[0])).clip(0, 1), 
                                        tpldf['yaw'])



            # ----------- Player velocities ------------
            def adjust_velocity(velocity, random_filter):
                new_velocity = np.random.normal(velocity, velocity / 1.2)
                return np.where(random_filter, 
                                np.clip(new_velocity, 0, velocity),
                                velocity)

            random_filter = np.random.rand(tpldf.shape[0]) < probability

            tpldf['velocity_X'] = adjust_velocity(tpldf['velocity_X'], random_filter)
            tpldf['velocity_Y'] = adjust_velocity(tpldf['velocity_Y'], random_filter)



            # ----------- Player health ------------
            tpldf['health'] = np.where(random_filter, 
                                        (tpldf['health'] + np.random.normal(0, 0.07, tpldf.shape[0])).clip(0, 1).round(2),
                                        tpldf['health'])

            tpldf.loc[(tpldf['health'] == 0) & (tpldf['is_alive'] == 1), 'is_alive'] = 0

            # ----------- Player armor ------------
            tpldf['armor_value'] = np.where(random_filter, 
                                        (tpldf['armor_value'] + np.random.normal(0, 0.07, tpldf.shape[0])).clip(0, 1).round(2),
                                        tpldf['armor_value'])



            # ----------- Player is flashed ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['flash_duration'] = np.where(random_filter,
                                            (tpldf['flash_duration'] + np.random.normal(0, 0.5, tpldf.shape[0])).clip(0, 1).round(4),
                                            tpldf['flash_duration'])



            # ----------- Player active weapon magazine ammo left % ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            ammo_filter = (tpldf['active_weapon_Knife'] == 0) & (tpldf['active_weapon_C4'] == 0) & (tpldf['active_weapon_Taser'] == 0)

            tpldf['active_weapon_magazine_ammo_left_%'] = np.where(ammo_filter & random_filter, 
                                            (tpldf['active_weapon_magazine_ammo_left_%'] + np.random.normal(0, 0.1, tpldf.shape[0])).clip(0, 1), 
                                            tpldf['active_weapon_magazine_ammo_left_%'])



            # ----------- Player is shooting ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['is_shooting'] = np.where(random_filter,
                                                1 - tpldf['is_shooting'],  # Flipping the value
                                                tpldf['is_shooting'])

            ammo_filter_shooting = (tpldf['is_shooting'] == 1) & (tpldf['active_weapon_magazine_ammo_left_%'] == 1)
            tpldf['active_weapon_magazine_ammo_left_%'] = np.where(ammo_filter_shooting,
                                                                    np.clip(tpldf['active_weapon_magazine_ammo_left_%'] - np.random.uniform(0.02, 0.12, tpldf.shape[0]), 0, 1),
                                                                    tpldf['active_weapon_magazine_ammo_left_%'])



            # ----------- Player is spotted ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['is_spotted'] = np.where(random_filter,
                                            1 - tpldf['is_spotted'],  # Flipping the value
                                            tpldf['is_spotted'])



            # ----------- Player is walking ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['is_walking'] = np.where(random_filter,
                                            1 - tpldf['is_walking'],  # Flipping the value
                                            tpldf['is_walking'])



            # ----------- Player is reloading ------------
            reloading_filter = np.random.rand(tpldf.shape[0]) < probability
            tpldf['is_reloading'] = np.where(reloading_filter,
                                                1 - tpldf['is_reloading'],  # Flipping the value
                                                tpldf['is_reloading'])

            tpldf['is_shooting'] = np.where(tpldf['is_reloading'] == 1, 0, tpldf['is_shooting'])



            # ----------- Player is scoped ------------
            random_filter = np.random.rand(tpldf.shape[0]) < probability
            weapon_columns = ['active_weapon_AWP', 'active_weapon_SSG 08', 'active_weapon_SG 553', 
                            'active_weapon_G3SG1', 'active_weapon_AUG', 'active_weapon_SCAR-20']

            tpldf['is_scoped'] = np.where(tpldf[weapon_columns].sum(axis=1) > 0,  # Check if any of the weapons are active
                                            np.where(random_filter, 1 - tpldf['is_scoped'], tpldf['is_scoped']),
                                            0)

            tpldf['zoom_lvl'] = np.where(tpldf['is_scoped'] == 1, 1, tpldf['zoom_lvl'])
            tpldf['zoom_lvl'] = np.where(tpldf['is_scoped'] == 0, 0, tpldf['zoom_lvl'])


            # ----------- Player is defusing ------------
            tpldf['is_defusing'] = np.where(tpldf['is_defusing'] == 1,
                                                np.where(np.random.rand(tpldf.shape[0]) < 0.2, 0, 1),
                                                tpldf['is_defusing'])



            # ----------- Player has C4 ------------
            tpldf['inventory_C4'] = np.where(tpldf['inventory_C4'] == 1,
                                                    np.where(np.random.rand(tpldf.shape[0]) < probability, 0, 1),
                                                    tpldf['inventory_C4'])



            # ----------- Switch active weapon ------------
            inventory_columns = ['inventory_C4', 'inventory_Taser', 'inventory_USP-S', 'inventory_P2000', 'inventory_Glock-18', 'inventory_Dual Berettas', 'inventory_P250', 'inventory_Tec-9', 'inventory_CZ75 Auto', 'inventory_Five-SeveN', 'inventory_Desert Eagle', 'inventory_R8 Revolver', 'inventory_MAC-10', 'inventory_MP9', 'inventory_MP7', 'inventory_MP5-SD', 'inventory_UMP-45', 'inventory_PP-Bizon', 'inventory_P90', 'inventory_Nova', 'inventory_XM1014', 'inventory_Sawed-Off', 'inventory_MAG-7', 'inventory_M249', 'inventory_Negev', 'inventory_FAMAS', 'inventory_Galil AR', 'inventory_AK-47', 'inventory_M4A4', 'inventory_M4A1-S', 'inventory_SG 553', 'inventory_AUG', 'inventory_SSG 08', 'inventory_AWP', 'inventory_G3SG1', 'inventory_SCAR-20', 'inventory_HE Grenade', 'inventory_Flashbang', 'inventory_Smoke Grenade', 'inventory_Incendiary Grenade', 'inventory_Molotov', 'inventory_Decoy Grenade']
            active_weapon_columns = ['active_weapon_C4', 'active_weapon_Knife', 'active_weapon_Taser', 'active_weapon_USP-S', 'active_weapon_P2000', 'active_weapon_Glock-18', 'active_weapon_Dual Berettas', 'active_weapon_P250', 'active_weapon_Tec-9', 'active_weapon_CZ75 Auto', 'active_weapon_Five-SeveN', 'active_weapon_Desert Eagle', 'active_weapon_R8 Revolver', 'active_weapon_MAC-10', 'active_weapon_MP9', 'active_weapon_MP7', 'active_weapon_MP5-SD', 'active_weapon_UMP-45', 'active_weapon_PP-Bizon', 'active_weapon_P90', 'active_weapon_Nova', 'active_weapon_XM1014', 'active_weapon_Sawed-Off', 'active_weapon_MAG-7', 'active_weapon_M249', 'active_weapon_Negev', 'active_weapon_FAMAS', 'active_weapon_Galil AR', 'active_weapon_AK-47', 'active_weapon_M4A4', 'active_weapon_M4A1-S', 'active_weapon_SG 553', 'active_weapon_AUG', 'active_weapon_SSG 08', 'active_weapon_AWP', 'active_weapon_G3SG1', 'active_weapon_SCAR-20', 'active_weapon_HE Grenade', 'active_weapon_Flashbang', 'active_weapon_Smoke Grenade', 'active_weapon_Incendiary Grenade', 'active_weapon_Molotov', 'active_weapon_Decoy Grenade']

            def get_inventory_weapons(row):
                return [col for col in inventory_columns if row[col] == 1]

            def get_active_weapon(row):
                return [col for col in active_weapon_columns if row[col] == 1]

            def switch_weapon(row):

                inventory_weapons = get_inventory_weapons(row)
                active_weapon = get_active_weapon(row)

                if len(inventory_weapons) == 0:
                    return row

                if len(active_weapon) == 0:
                    return row
                if len(active_weapon) == 1:
                    active_weapon = active_weapon[0]

                # Random roll
                if np.random.rand() < 0.5:
                    return row

                else:
                    new_active_weapon = random.choice(inventory_weapons)
                    row[active_weapon] = 0
                    row[new_active_weapon] = 1

                    return row
                
            tpldf = tpldf.apply(switch_weapon, axis=1)


            sample['player'].x = torch.tensor(tpldf.values, dtype=torch.float32)

        return samples
    


    # Update the player-map edges of the graph
    def _update_player_map_edges(self, samples: list):
        """
        Update the player-map edges of the graph.
        Parameters:
        - samples: the list of HeteroData samples to update.
        - map_name: the name of the map.
        - normalized: whether the input graph is normalized.
        """

        # Update the player-map edges for each sample
        for sample in samples:
            
            player_closest_to_map = []

            for i in range(10):
                # Get the map node with the closest coordinates to the i-th player
                player_coords = sample.x_dict['player'][i, 0:3].numpy()
                map_coords = sample.x_dict['map'][:, 1:4].numpy()
                distances = np.linalg.norm(map_coords - player_coords, axis=1)
                closest_node = np.argmin(distances)
                player_closest_to_map.append(closest_node)

            del sample['player', 'closest_to', 'map']
            sample['player', 'closest_to', 'map'].edge_index = torch.tensor([list(range(10)), player_closest_to_map], dtype=torch.int16)

        return samples
    


    # Update map node burning and smoked values
    def _update_map_node_burning_smoked_values(self, samples: list):
        """
        Update the map node burning and smoked values of the graph.
        Parameters:
        - samples: the list of HeteroData samples to update.
        """

        # Update the map node burning and smoked values for each sample
        for sample in samples:
            
            mdf = pd.DataFrame(sample['map'].x, columns = ['posid', 'X', 'Y', 'Z', 'is_contact', 'is_bombsite', 'is_bomb_planted_near', 'is_burning', 'is_smoked'])

            # ---------- Map molotovs ------------
            random_filter = np.random.rand(mdf.shape[0]) < 0.25
            burning_filter = (mdf['is_burning'] == 1)

            mdf['is_burning'] = np.where(random_filter & burning_filter,
                                            1 - mdf['is_burning'],  # Flipping the value
                                            mdf['is_burning'])

            # ---------- Map smokes ------------
            random_filter = np.random.rand(mdf.shape[0]) < 0.25
            smoked_filter = (mdf['is_smoked'] == 1)

            mdf['is_smoked'] = np.where(random_filter & smoked_filter,
                                            1 - mdf['is_smoked'],  # Flipping the value
                                            mdf['is_smoked'])

            sample['map'].x = torch.tensor(mdf.values, dtype=torch.float32)

        return samples
    


    # Update the y values of the graph
    def _update_y_values(
            self, 
            samples: list,
            scaling_dict_current_player_equip_value_max: int = 8450, 
            scaling_dict_CT_equip_value_max: int = 35100,
            scaling_dict_T_equip_value_max: int = 31600,):
        """
        Update the y values of the graph.
        Parameters:
        - samples: the list of HeteroData samples to update.
        """

        # Update the y values for each sample
        for sample in samples:

            tpldf = pd.DataFrame(sample.x_dict['player'].numpy(), columns=self.player_columns)
            
            sample.y['CT_alive_num'] = tpldf.iloc[0:5]['is_alive'].sum()
            sample.y['T_alive_num'] = tpldf.iloc[5:10]['is_alive'].sum()

            sample.y['CT_total_hp'] = tpldf.iloc[0:5]['health'].sum()
            sample.y['T_total_hp'] = tpldf.iloc[5:10]['health'].sum()

            sample.y['CT_equipment_value'] = round(sum(tpldf.iloc[0:5]['current_equip_value'] * scaling_dict_current_player_equip_value_max)) / scaling_dict_CT_equip_value_max
            sample.y['T_equipment_value'] = round(sum(tpldf.iloc[5:10]['current_equip_value'] * scaling_dict_current_player_equip_value_max)) / scaling_dict_T_equip_value_max

        return samples
