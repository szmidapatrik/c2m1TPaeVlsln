from awpy import Demo
import pandas as pd
import numpy as np
import random


class CS2_TabularSnapshots:

    # INPUT
    # Folder path constants
    MATCH_PATH = None
    PLAYER_STATS_DATA_PATH = None
    MISSING_PLAYER_STATS_DATA_PATH = None

    
    # Optional variables
    ticks_per_second = 1
    numerical_match_id = None
    num_permutations_per_round = 1
    build_dictionary = True

    # Other variables
    __nth_tick__ = 1



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    def process_match(
        self,
        match_path: str,
        player_stats_data_path: str, 
        missing_player_stats_data_path: str,

        ticks_per_second: int = 1,
        numerical_match_id: int = None,
        num_permutations_per_round: int = 1,
        build_dictionary: bool = True,
    ):
        """
        Formats the match data and creates the tabular game-snapshot dataset. Parameters:
            - match_path: name of the match file,
            - player_stats_data_path: path of the player statistics data,
            - missing_player_stats_data_path: path of the missing player statistics data,

            - ticks_per_second (optional): how many ticks should be returned for each second. Values: 1, 2, 4, 8, 16, 32 and 64. Default is 1.
            - numerical_match_id (optional): numerical match id to add to the dataset. If value is None, no numerical match id will be added. Default is None.
            - num_permutations_per_round (optional): number of different player permutations to create for the snapshots per round. Default is 1.
            - build_dictionary (optional): whether to build and return a dictionary with the min and max column values. Default is True.
        """

        # INPUT
        self.MATCH_PATH = match_path
        self.PLAYER_STATS_DATA_PATH = player_stats_data_path
        self.MISSING_PLAYER_STATS_DATA_PATH = missing_player_stats_data_path

        # Other variables
        self.ticks_per_second = ticks_per_second
        self.numerical_match_id = numerical_match_id
        self.num_permutations_per_round = num_permutations_per_round

        # 0. Ticks per second operations
        self.__PREP_ticks_per_second_operations__()

        # 1.
        ticks, kills, rounds, bomb, damages, smokes, infernos = self._INIT_dataframes()

        # 2.
        pf = self._PLAYER_ingame_stats(ticks, kills, rounds, damages)

        # 3.
        pf = self._PLAYER_inventory(pf)
        
        # 4.
        pf = self._PLAYER_active_weapons(pf)

        # 5.
        players = self._PLAYER_player_datasets(pf)

        # 6.
        players = self._PLAYER_hltv_statistics(players)

        # 7.
        tabular_df = self._TABULAR_initial_dataset(players, rounds, self.MATCH_PATH)

        # 8.
        tabular_df = self._TABULAR_bomb_info(tabular_df, bomb)

        # 9.
        tabular_df = self._TABULAR_INFERNO_bombsite_3x3_split(tabular_df)

        # 10.
        tabular_df = self._TABULAR_smokes_and_molotovs(tabular_df, smokes, infernos)

        # 11.
        if self.numerical_match_id is not None:
            tabular_df = self._TABULAR_numerical_match_id(tabular_df)

        # 12.
        if num_permutations_per_round > 1:
            tabular_df = self._TABULAR_player_permutation(tabular_df, self.num_permutations_per_round)
            
        # 13.
        tabular_df = self._TABULAR_refactor_player_columns(tabular_df)

        # 14.
        if build_dictionary:
            tabular_df_dict = self._FINAL_build_dictionary(tabular_df)

        # 15.
        self._FINAL_free_memory(ticks, kills, rounds, bomb, damages, smokes, infernos)

        # Return
        if build_dictionary:
            return tabular_df, tabular_df_dict
        else:
            return tabular_df



    def impute_match(
        self, 
        df, 
        method='zero'
    ):
        """
        Imputes missing values in the dataset. Parameters:
            - df: the dataset to be imputed.
            - method (optional): the method to be used for imputation. Can be 'zero'. Default is 'zero'.
        """

        # Method: zero
        if method == 'zero':
            df = df.infer_objects().fillna(0)

        return df



    def noramlize_match(
        self,
        df,
        position_scaler,
        dictionary,
    ):
        """
        Normalizes the dataset. Parameters:
            - df: the dataset to be normalized.
            - position_scaler: the scaler to be used for the positional columns. Often it is the map node scaling model.
            - dictionary: the dictionary with the min and max values of each column.
        """

        for col in df.columns:
            
            # Format column name
            dict_column_name = col[3:] if col.startswith('CT') else col[2:]

            # Different normalization methods for different columns
            # Position columns are normalized using the position_scaler - skip them
            if dict_column_name in ['_X', '_Y', '_Z', 'bomb_X', 'bomb_Y', 'bomb_Z']:
                continue

            # Skip the state-describint boolean columns (values are already 0 or 1)
            if dict_column_name.startswith('_is'):
                continue

            # Skip the inventory columns (values are already 0 or 1)
            if dict_column_name.startswith('_inventory'):
                continue

            # Skip the active weapon columns (values are already 0 or 1)
            if dict_column_name.startswith('_active_weapon'):
                continue

            # TODO: active_weapon_ammo - simple normalization is not enough, as the values are not in the same range (weapon dependent)

            # TODO: total_ammo_left - simple normalization is not enough, as the values are not in the same range (weapon dependent)

            # Transform other columns
            dict_values = dictionary.loc[dictionary['column'] == dict_column_name]
            df[col] = (df[col] - dict_values['min']) / (dict_values['max'] - dict_values['min']) 

        return df

    # --------------------------------------------------------------------------------------------

    # 0. Ticks per second operations
    def __PREP_ticks_per_second_operations__(self):
        
        # Check if the ticks_per_second is valid
        if self.ticks_per_second not in [1, 2, 4, 8, 16, 32, 64]:
            raise ValueError("Invalid ticks_per_second value. Please choose one of the following: 1, 2, 4, 8, 16, 32 or 64.")
        
        # Set the nth_tick value (the type must be integer)
        self.__nth_tick__ = int(64 / self.ticks_per_second)

    # 15. Free memory
    def _FINAL_free_memory(self, ticks, kills, rounds, bomb, damages, smokes, infernos):
        del ticks
        del kills
        del rounds
        del bomb
        del damages
        del smokes
        del infernos

    # --------------------------------------------------------------------------------------------

    # 1. Get needed dataframes
    def _INIT_dataframes(self):

        player_cols = [
            'X',
            'Y',
            'Z',
            'health',
            'score',
            'mvps',
            'is_alive',
            'balance',
            'inventory',
            'life_state',
            'pitch',
            'yaw',
            'armor',
            'has_defuser',
            'has_helmet',
            'player_name',
            'start_balance',
            'total_cash_spent',
            'cash_spent_this_round',
            'move_collide',
            'move_type',
            'team_num',
            'jump_velo',
            'fall_velo',
            'in_crouch',
            'crouch_state',
            'ducked',
            'ducking',
            'in_duck_jump',
            'spotted',
            'approximate_spotted_by',
            'time_last_injury',
            'player_state',
            'passive_items',
            'is_scoped',
            'is_walking',
            'resume_zoom',
            'is_defusing',
            'in_bomb_zone',
            'move_state',
            'which_bomb_zone',
            'in_hostage_rescue_zone',
            'stamina',
            'direction',
            'armor_value',
            'velo_modifier',
            'flash_duration',
            'flash_max_alpha',
            'round_start_equip_value',
            'current_equip_value',
            'velocity',
            'velocity_X',
            'velocity_Y',
            'velocity_Z',
            'FIRE',
        ]
        other_cols = [
            'num_player_alive_ct',
            'num_player_alive_t',
            'ct_losing_streak',
            't_losing_streak',
            'active_weapon_name',
            'active_weapon_ammo',
            'total_ammo_left',
            'is_in_reload',
            'alive_time_total',
            'is_bomb_dropped'
        ]

        match = Demo(path=self.MATCH_PATH, player_props=player_cols, other_props=other_cols)

        # Read dataframes
        ticks = match.ticks
        kills = match.kills
        rounds = match.rounds
        bomb = match.bomb
        damages = match.damages
        smokes = match.smokes
        infernos = match.infernos


        # Filter columns
        rounds = rounds[['round', 'freeze_end', 'end', 'winner']]
        ticks = ticks[['tick', 'round', 'team_name', 'name',
                       'X', 'Y', 'Z', 'pitch', 'yaw', 'velocity_X', 'velocity_Y', 'velocity_Z', 'inventory',
                       'health', 'armor_value', 'active_weapon_name', 'active_weapon_ammo', 'total_ammo_left',
                       'is_alive', 'in_crouch', 'ducking', 'in_duck_jump', 'is_walking', 'spotted', 'is_scoped', 'is_defusing', 'is_in_reload',
                       'flash_duration', 'in_bomb_zone', 'balance', 'current_equip_value', 'round_start_equip_value',
                       'ct_losing_streak', 't_losing_streak', 'is_bomb_dropped', 'FIRE'
                ]]
        
        ticks = ticks.rename(columns={
            'in_crouch'     : 'is_crouching',
            'ducking'       : 'is_ducking',
            'in_duck_jump'  : 'is_duck_jumping',
            'is_walking'    : 'is_walking',
            'spotted'       : 'is_spotted',
            'is_in_reload'  : 'is_reloading',
            'in_bomb_zone'  : 'is_in_bombsite',
            'FIRE'          : 'is_shooting'
        })
        
        return ticks, kills, rounds, bomb, damages, smokes, infernos



    # 2. Calculate ingame player statistics
    def _PLAYER_ingame_stats(self, ticks, kills, rounds, damages):
    
        # Merge playerFrames with rounds
        pf = ticks.merge(rounds, on='round')

        # Format CT information
        pf['is_CT'] = pf.apply(lambda x: 1 if x['team_name'] == 'CT' else 0, axis=1)
        del pf['team_name']

        # Kill stats
        pf['stat_kills'] = 0
        pf['stat_HS_kills'] = 0
        pf['stat_opening_kills'] = 0

        # Death stats
        pf['stat_deaths'] = 0
        pf['stat_opening_deaths'] = 0

        # Assist stats
        pf['stat_assists'] = 0
        pf['stat_flash_assists'] = 0

        # Damage stats
        pf['stat_damage'] = 0
        pf['stat_weapon_damage'] = 0
        pf['stat_nade_damage'] = 0

        # Setting kill-stats
        for _, row in kills.iterrows():

            # Kills
            pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_kills'] += 1
            # HS-kills
            if row['headshot']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_HS_kills'] += 1
            # Opening-kills
            if row['tick'] == kills.loc[kills['round'] == row['round']].iloc[0]['tick']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_opening_kills'] += 1


            # Deaths
            pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['victim_name']), 'stat_deaths'] += 1
            # Opening deaths
            if row['tick'] == kills.loc[kills['round'] == row['round']].iloc[0]['tick']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['victim_name']), 'stat_opening_deaths'] += 1


            # Assists
            if pd.notna(row['assister_name']):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['assister_name']), 'stat_assists'] += 1

            # Flash assists
            if row['assistedflash']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['assister_name']), 'stat_flash_assists'] += 1


        # Setting damage-stats
        for _, row in damages.iterrows():

            # All Damage
            if (row['attacker_team_name'] != row['victim_team_name']):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_damage'] += row['dmg_health_real']

            # Weapon Damage
            if (row['attacker_team_name'] != row['victim_team_name']) and (row['weapon'] not in ['inferno', 'molotov', 'hegrenade', 'flashbang', 'smokegrenade']):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_weapon_damage'] += row['dmg_health_real']

            # Nade Damage
            if (row['attacker_team_name'] != row['victim_team_name']) and (row['weapon'] in ['inferno', 'molotov', 'hegrenade', 'flashbang', 'smokegrenade']):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attacker_name']), 'stat_nade_damage'] += row['dmg_health_real']

        # Calculate other stats
        pf['stat_survives'] = pf['round'] - pf['stat_deaths']
        pf['stat_KPR'] = pf['stat_kills'] / pf['round']
        pf['stat_ADR'] = pf['stat_damage'] / pf['round']
        pf['stat_DPR'] = pf['stat_deaths'] / pf['round']
        pf['stat_HS%'] = pf['stat_HS_kills'] / pf['stat_kills']
        pf['stat_SPR'] = pf['stat_survives'] / pf['round']
            
        return pf
    


        # 3. Handle active weapon column
    
    
    
    # 3. Inventory
    def _PLAYER_inventory(self, pf):

        # Inventory weapons
        inventory_weapons = [
            # Other
            'inventory_C4', 'inventory_Taser',
            # Pistols
            'inventory_USP-S', 'inventory_P2000', 'inventory_Glock-18', 'inventory_Dual Berettas', 'inventory_P250', 'inventory_Tec-9', 'inventory_CZ75 Auto', 'inventory_Five-SeveN', 'inventory_Desert Eagle',
            # SMGs
            'inventory_MAC-10', 'inventory_MP9', 'inventory_MP7', 'inventory_MP5-SD', 'inventory_UMP-45', 'inventory_PP-Bizon', 'inventory_P90',
            # Heavy
            'inventory_Nova', 'inventory_XM1014', 'inventory_Sawed-Off', 'inventory_MAG-7', 'inventory_M249', 'inventory_Negev',
            # Rifles
            'inventory_FAMAS', 'inventory_Galil AR', 'inventory_AK-47', 'inventory_M4A4', 'inventory_M4A1-S', 'inventory_SG 553', 'inventory_AUG', 'inventory_SSG 08', 'inventory_AWP', 'inventory_G3SG1', 'inventory_SCAR-20',
            # Grenades
            'inventory_HE Grenade', 'inventory_Flashbang', 'inventory_Smoke Grenade', 'inventory_Incendiary Grenade', 'inventory_Molotov', 'inventory_Decoy Grenade'
        ]

        # Create dummie cols
        for col in inventory_weapons:
            pf[col] = pf['inventory'].apply(lambda x: 1 if col.replace('inventory_', '') in x else 0)

        return pf



    # 4. Handle active weapon column
    def _PLAYER_active_weapons(self, pf):

        # If the actifWeapon column value contains the word knife, set the activeWeapon column to 'Knife'
        pf['active_weapon_name'] = pf['active_weapon_name'].fillna('')
        pf['active_weapon_name'] = pf['active_weapon_name'].apply(lambda x: 'Knife' if 'knife' in str.lower(x) else x)
    
        # Active weapons
        active_weapons = [
            # Other
            'active_weapon_C4', 'active_weapon_Knife', 'active_weapon_Taser',
            # Pistols
            'active_weapon_USP-S', 'active_weapon_P2000', 'active_weapon_Glock-18', 'active_weapon_Dual Berettas', 'active_weapon_P250', 'active_weapon_Tec-9', 'active_weapon_CZ75 Auto', 'active_weapon_Five-SeveN', 'active_weapon_Desert Eagle',
            # SMGs
            'active_weapon_MAC-10', 'active_weapon_MP9', 'active_weapon_MP7', 'active_weapon_MP5-SD', 'active_weapon_UMP-45', 'active_weapon_PP-Bizon', 'active_weapon_P90',
            # Heavy
            'active_weapon_Nova', 'active_weapon_XM1014', 'active_weapon_Sawed-Off', 'active_weapon_MAG-7', 'active_weapon_M249', 'active_weapon_Negev',
            # Rifles
            'active_weapon_FAMAS', 'active_weapon_Galil AR', 'active_weapon_AK-47', 'active_weapon_M4A4', 'active_weapon_M4A1-S', 'active_weapon_SG 553', 'active_weapon_AUG', 'active_weapon_SSG 08', 'active_weapon_AWP', 'active_weapon_G3SG1', 'active_weapon_SCAR-20',
            # Grenades
            'active_weapon_HE Grenade', 'active_weapon_Flashbang', 'active_weapon_Smoke Grenade', 'active_weapon_Incendiary Grenade', 'active_weapon_Molotov', 'active_weapon_Decoy Grenade'
        ]

        # Create dummie cols
        df_dummies = pd.get_dummies(pf['active_weapon_name'], prefix="active_weapon",drop_first=False)
        dummies = pd.DataFrame()
        for col in active_weapons:
            if col not in df_dummies.columns:
                dummies[col] = np.zeros(len(df_dummies))
            else:
                dummies[col] = df_dummies[col]
        
        dummies = dummies*1
        pf = pf.merge(dummies, left_index = True, right_index = True, how = 'left')
        
        return pf
    


    # 5. Create player dataset
    def _PLAYER_player_datasets(self, pf):
    
        startAsCTPlayerNames = pf[(pf['is_CT'] == True)  & (pf['round'] == 1)]['name'].drop_duplicates().tolist()
        startAsTPlayerNames  = pf[(pf['is_CT'] == False) & (pf['round'] == 1)]['name'].drop_duplicates().tolist()

        players = {}

        # Team 1: start on CT side
        players[0] = pf[pf['name'] == startAsCTPlayerNames[0]].iloc[::self.__nth_tick__].copy()
        players[1] = pf[pf['name'] == startAsCTPlayerNames[1]].iloc[::self.__nth_tick__].copy()
        players[2] = pf[pf['name'] == startAsCTPlayerNames[2]].iloc[::self.__nth_tick__].copy()
        players[3] = pf[pf['name'] == startAsCTPlayerNames[3]].iloc[::self.__nth_tick__].copy()
        players[4] = pf[pf['name'] == startAsCTPlayerNames[4]].iloc[::self.__nth_tick__].copy()

        # Team 2: start on T side
        players[5] = pf[pf['name'] == startAsTPlayerNames[0]].iloc[::self.__nth_tick__].copy()
        players[6] = pf[pf['name'] == startAsTPlayerNames[1]].iloc[::self.__nth_tick__].copy()
        players[7] = pf[pf['name'] == startAsTPlayerNames[2]].iloc[::self.__nth_tick__].copy()
        players[8] = pf[pf['name'] == startAsTPlayerNames[3]].iloc[::self.__nth_tick__].copy()
        players[9] = pf[pf['name'] == startAsTPlayerNames[4]].iloc[::self.__nth_tick__].copy()
        
        return players
    


    # 6. Insert universal player statistics into player dataset
    def __EXT_insert_columns_into_player_dataframes__(self, stat_df, players_df):
        for col in stat_df.columns:
            if col != 'player_name':
                players_df[col] = stat_df.loc[stat_df['player_name'] == players_df['name'].iloc[0]][col].iloc[0]
        return players_df

    def _PLAYER_hltv_statistics(self, players):
        # Needed columns
        needed_stats = ['player_name', 'rating_2.0', 'DPR', 'KAST', 'Impact', 'ADR', 'KPR','total_kills', 'HS%', 'total_deaths', 'KD_ratio', 'dmgPR',
        'grenade_dmgPR', 'maps_played', 'saved_by_teammatePR', 'saved_teammatesPR','opening_kill_rating', 'team_W%_after_opening',
        'opening_kill_in_W_rounds', 'rating_1.0_all_Career', 'clutches_1on1_ratio', 'clutches_won_1on1', 'clutches_won_1on2', 'clutches_won_1on3', 'clutches_won_1on4', 'clutches_won_1on5']
        
        stats = pd.read_csv(self.PLAYER_STATS_DATA_PATH).drop_duplicates()

        try:
            stats = stats[needed_stats]

        # If clutches_1on1_ratio column is missing, calculate it here
        except:
            stats['clutches_1on1_ratio'] = stats['clutches_won_1on1'] / stats['clutches_lost_1on1']
            stats['clutches_1on1_ratio'] = stats['clutches_1on1_ratio'].fillna(0)
            stats = stats[needed_stats]

        # Stats dataframe basic formatting
        for col in stats.columns:
            if col != 'player_name':
                stats[col] = stats[col].astype('float32')
                stats.rename(columns={col: "hltv_" + col}, inplace=True)
        
        # Merge stats with players
        for idx in range(0,len(players)):
            # If the stats dataframe contains the player related informations, do the merge
            if len(stats.loc[stats['player_name'] == players[idx]['name'].iloc[0]]) == 1:
                players[idx] = self.__EXT_insert_columns_into_player_dataframes__(stats, players[idx])

            # If the stats dataframe does not contain the player related informations, check if the missing_players_df contains the player
            else:

                mpdf = pd.read_csv(self.MISSING_PLAYER_STATS_DATA_PATH)
                
                try:
                    mpdf = mpdf[needed_stats]
                    
                # If clutches_1on1_ratio column is missing, calculate it here
                except:
                    mpdf['clutches_1on1_ratio'] = mpdf['clutches_won_1on1'] / mpdf['clutches_lost_1on1']
                    mpdf['clutches_1on1_ratio'] = mpdf['clutches_1on1_ratio'].fillna(0)
                    mpdf = mpdf[needed_stats]
                
                for col in mpdf.columns:
                    if col != 'player_name':
                        mpdf[col] = mpdf[col].astype('float32')
                        mpdf.rename(columns={col: "hltv_" + col}, inplace=True)
                        
                # If the missing_players_df contains the player related informations, do the merge
                if len(mpdf.loc[mpdf['player_name'] == players[idx]['name'].iloc[0]]) == 1:
                    players[idx] = self.__EXT_insert_columns_into_player_dataframes__(mpdf, players[idx])

                # Else get imputed values for the player from missing_players_df and do the merge
                else:
                    first_anonim_pro_index = mpdf.index[mpdf['player_name'] == 'anonim_pro'].min()
                    mpdf.at[first_anonim_pro_index, 'player_name'] = players[idx]['name'].iloc[0]
                    players[idx] = self.__EXT_insert_columns_into_player_dataframes__(mpdf, players[idx])
                    
                    # Reverse the column renaming - remove the 'hltv_' prefix
                    for col in mpdf.columns:
                        if col.startswith('hltv_'):
                            new_col = col[len('hltv_'):]
                            mpdf.rename(columns={col: new_col}, inplace=True)

                    mpdf.to_csv(self.MISSING_PLAYER_STATS_DATA_PATH, index=False)
            
        return players
    


    # 7. Create tabular dataset - first version (1 row - 1 graph)
    def __EXT_calculate_ct_equipment_value__(self, row):
        if row['player0_is_CT']:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __EXT_calculate_t_equipment_value__(self, row):
        if row['player0_is_CT'] == False:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __EXT_calculate_ct_total_hp__(self, row):
        if row['player0_is_CT']:
            return row[['player0_health','player1_health','player2_health','player3_health','player4_health']].sum()
        else:
            return row[['player5_health','player6_health','player7_health','player8_health','player9_health']].sum()

    def __EXT_calculate_t_total_hp__(self, row):
        if row['player0_is_CT'] == False:
            return row[['player0_health','player1_health','player2_health','player3_health','player4_health']].sum()
        else:
            return row[['player5_health','player6_health','player7_health','player8_health','player9_health']].sum()

    def __EXT_calculate_ct_alive_num__(self, row):
        if row['player0_is_CT']:
            return row[['player0_is_alive','player1_is_alive','player2_is_alive','player3_is_alive','player4_is_alive']].sum()
        else:
            return row[['player5_is_alive','player6_is_alive','player7_is_alive','player8_is_alive','player9_is_alive']].sum()

    def __EXT_calculate_t_alive_num__(self, row):
        if row['player0_is_CT'] == False:
            return row[['player0_is_alive','player1_is_alive','player2_is_alive','player3_is_alive','player4_is_alive']].sum()
        else:
            return row[['player5_is_alive','player6_is_alive','player7_is_alive','player8_is_alive','player9_is_alive']].sum()

    def __EXT_delete_useless_columns__(self, graph_data):

        del graph_data['player0_equi_val_alive']
        del graph_data['player1_equi_val_alive']
        del graph_data['player2_equi_val_alive']
        del graph_data['player3_equi_val_alive']
        del graph_data['player4_equi_val_alive']
        del graph_data['player5_equi_val_alive']
        del graph_data['player6_equi_val_alive']
        del graph_data['player7_equi_val_alive']
        del graph_data['player8_equi_val_alive']
        del graph_data['player9_equi_val_alive']
        
        del graph_data['player0_freeze_end']
        del graph_data['player1_freeze_end']
        del graph_data['player2_freeze_end']
        del graph_data['player3_freeze_end']
        del graph_data['player4_freeze_end']
        del graph_data['player5_freeze_end']
        del graph_data['player6_freeze_end']
        del graph_data['player7_freeze_end']
        del graph_data['player8_freeze_end']
        del graph_data['player9_freeze_end']
        
        del graph_data['player0_end']
        del graph_data['player1_end']
        del graph_data['player2_end']
        del graph_data['player3_end']
        del graph_data['player4_end']
        del graph_data['player5_end']
        del graph_data['player6_end']
        del graph_data['player7_end']
        del graph_data['player8_end']
        del graph_data['player9_end']
        
        del graph_data['player0_winner']
        del graph_data['player1_winner']
        del graph_data['player2_winner']
        del graph_data['player3_winner']
        del graph_data['player4_winner']
        del graph_data['player5_winner']
        del graph_data['player6_winner']
        del graph_data['player7_winner']
        del graph_data['player8_winner']
        del graph_data['player9_winner']

        del graph_data['player1_ct_losing_streak']
        del graph_data['player2_ct_losing_streak']
        del graph_data['player3_ct_losing_streak']
        del graph_data['player4_ct_losing_streak']
        del graph_data['player5_ct_losing_streak']
        del graph_data['player6_ct_losing_streak']
        del graph_data['player7_ct_losing_streak']
        del graph_data['player8_ct_losing_streak']
        del graph_data['player9_ct_losing_streak']

        del graph_data['player1_t_losing_streak']
        del graph_data['player2_t_losing_streak']
        del graph_data['player3_t_losing_streak']
        del graph_data['player4_t_losing_streak']
        del graph_data['player5_t_losing_streak']
        del graph_data['player6_t_losing_streak']
        del graph_data['player7_t_losing_streak']
        del graph_data['player8_t_losing_streak']
        del graph_data['player9_t_losing_streak']

        del graph_data['player1_is_bomb_dropped']
        del graph_data['player2_is_bomb_dropped']
        del graph_data['player3_is_bomb_dropped']
        del graph_data['player4_is_bomb_dropped']
        del graph_data['player5_is_bomb_dropped']
        del graph_data['player6_is_bomb_dropped']
        del graph_data['player7_is_bomb_dropped']
        del graph_data['player8_is_bomb_dropped']
        del graph_data['player9_is_bomb_dropped']

        return graph_data

    def __EXT_calculate_time_remaining__(self, row):
        return 115.0 - ((row['tick'] - row['freeze_end']) / 64.0)

    def _TABULAR_initial_dataset(self, players, rounds, match_id):
        """
        Creates the first version of the dataset for the graph model.
        """

        # Copy players object
        graph_players = {}
        for idx in range(0,len(players)):
            graph_players[idx] = players[idx].copy()

        colsNotToRename = ['tick', 'round']

        # Rename columns except for tick, roundNum, seconds, floorSec
        for idx in range(0,len(graph_players)):
            
            for col in graph_players[idx].columns:
                if col not in colsNotToRename:
                    graph_players[idx].rename(columns={col: "player" + str(idx) + "_" + col}, inplace=True)

        # Create a graph dataframe to store all players in 1 row per second
        graph_data = graph_players[0].copy()

        # Merge dataframes
        for i in range(1, len(graph_players)):
            graph_data = graph_data.merge(graph_players[i], on=colsNotToRename)

        graph_data = graph_data.merge(rounds, on=['round'])
        graph_data['CT_wins'] = graph_data.apply(lambda x: 1 if (x['winner'] == 'CT') else 0, axis=1)

        graph_data['player0_equi_val_alive'] = graph_data['player0_current_equip_value'] * graph_data['player0_is_alive']
        graph_data['player1_equi_val_alive'] = graph_data['player1_current_equip_value'] * graph_data['player1_is_alive']
        graph_data['player2_equi_val_alive'] = graph_data['player2_current_equip_value'] * graph_data['player2_is_alive']
        graph_data['player3_equi_val_alive'] = graph_data['player3_current_equip_value'] * graph_data['player3_is_alive']
        graph_data['player4_equi_val_alive'] = graph_data['player4_current_equip_value'] * graph_data['player4_is_alive']
        graph_data['player5_equi_val_alive'] = graph_data['player5_current_equip_value'] * graph_data['player5_is_alive']
        graph_data['player6_equi_val_alive'] = graph_data['player6_current_equip_value'] * graph_data['player6_is_alive']
        graph_data['player7_equi_val_alive'] = graph_data['player7_current_equip_value'] * graph_data['player7_is_alive']
        graph_data['player8_equi_val_alive'] = graph_data['player8_current_equip_value'] * graph_data['player8_is_alive']
        graph_data['player9_equi_val_alive'] = graph_data['player9_current_equip_value'] * graph_data['player9_is_alive']

        graph_data['CT_alive_num'] = graph_data.apply(self.__EXT_calculate_ct_alive_num__, axis=1)
        graph_data['T_alive_num']  = graph_data.apply(self.__EXT_calculate_t_equipment_value__, axis=1)
        
        graph_data['CT_total_hp'] = graph_data.apply(self.__EXT_calculate_ct_total_hp__, axis=1)
        graph_data['T_total_hp']  = graph_data.apply(self.__EXT_calculate_t_total_hp__, axis=1)

        graph_data['CT_equipment_value'] = graph_data.apply(self.__EXT_calculate_ct_equipment_value__, axis=1)
        graph_data['T_equipment_value'] = graph_data.apply(self.__EXT_calculate_t_equipment_value__, axis=1)

        graph_data = graph_data.rename(columns={
            'player0_ct_losing_streak': 'CT_losing_streak', 
            'player0_t_losing_streak': 'T_losing_streak', 
            'player0_is_bomb_dropped': 'is_bomb_dropped',
        })

        graph_data = self.__EXT_delete_useless_columns__(graph_data)

        # Add time remaining column
        new_columns = pd.DataFrame({
            'time': 0.0,
        }, index=graph_data.index)
        graph_data = pd.concat([graph_data, new_columns], axis=1)
        graph_data['time'] = graph_data.apply(self.__EXT_calculate_time_remaining__, axis=1)

        # Create a DataFrame with a single column for match_id
        match_id_df = pd.DataFrame({'match_id': str(match_id)}, index=graph_data.index)
        graph_data_concatenated = pd.concat([graph_data, match_id_df], axis=1)

        return graph_data_concatenated


    # 8. Add bomb information to the dataset
    def __EXT_calculate_is_bomb_being_planted__(self, row):
        for i in range(0,10):
            if row['player{}_active_weapon_C4'.format(i)] == 1:
                if row['player{}_is_in_bombsite'.format(i)] == 1:
                    if row['player{}_is_shooting'.format(i)] == 1:
                        return 1
        return 0
    
    def __EXT_calculate_is_bomb_being_defused__(self, row):
        return row['player0_is_defusing'] + row['player1_is_defusing'] + row['player2_is_defusing'] + row['player3_is_defusing'] + row['player4_is_defusing'] + \
               row['player5_is_defusing'] + row['player6_is_defusing'] + row['player7_is_defusing'] + row['player8_is_defusing'] + row['player9_is_defusing']

    def _TABULAR_bomb_info(self, tabular_df, bombdf):

        new_columns = pd.DataFrame({
            'is_bomb_being_planted': 0,
            'is_bomb_being_defused': 0,
            'is_bomb_defused': 0,
            'is_bomb_planted_at_A_site': 0,
            'is_bomb_planted_at_B_site': 0,
            'plant_tick': 0,
            'bomb_X': 0.0,
            'bomb_Y': 0.0,
            'bomb_Z': 0.0
        }, index=tabular_df.index)

        tabular_df = pd.concat([tabular_df, new_columns], axis=1)

        tabular_df['is_bomb_being_planted'] = tabular_df.apply(self.__EXT_calculate_is_bomb_being_planted__, axis=1)
        tabular_df['is_bomb_being_defused'] = tabular_df.apply(self.__EXT_calculate_is_bomb_being_defused__, axis=1)

        for _, row in bombdf.iterrows():

            if (row['event'] == 'planted'):
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_planted_at_A_site'] = 1 if row['site'] == 'BombsiteA' else 0
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_planted_at_B_site'] = 1 if row['site'] == 'BombsiteB' else 0
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'bomb_X'] = row['X']
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'bomb_Y'] = row['Y']
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'bomb_Z'] = row['Z']
                tabular_df.loc[(tabular_df['round'] == row['round']), 'plant_tick'] = row['tick']

            if (row['event'] == 'defused'):
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_defused'] = 0
                tabular_df.loc[(tabular_df['round'] == row['round']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_defused'] = 1

        # Time remaining including the plant time
        tabular_df['remaining_time'] = tabular_df['time']
        tabular_df.loc[tabular_df['is_bomb_planted_at_A_site'] == 1, 'remaining_time'] = 40.0 - ((tabular_df['tick'] - tabular_df['plant_tick']) / 64.0)
        tabular_df.loc[tabular_df['is_bomb_planted_at_B_site'] == 1, 'remaining_time'] = 40.0 - ((tabular_df['tick'] - tabular_df['plant_tick']) / 64.0)


        return tabular_df



    # 9. Split the bombsites by 3x3 matrix for bomb position feature
    def __EXT_INFERNO_get_bomb_mx_coordinate__(self, row):
        # If bomb is planted on A
        if row['is_bomb_planted_at_A_site'] == 1:
                # 1st row
                if row['bomb_Y'] >= 650:
                    # 1st column
                    if row['bomb_X'] < 1900:
                        return 1
                    # 2nd column
                    if row['bomb_X'] >= 1900 and row['bomb_X'] < 2050:
                        return 2
                    # 3rd column
                    if row['bomb_X'] >= 2050:
                        return 3
                # 2nd row
                if row['bomb_Y'] < 650 and row['bomb_Y'] >= 325: 
                    # 1st column
                    if row['bomb_X'] < 1900:
                        return 4
                    # 2nd column
                    if row['bomb_X'] >= 1900 and row['bomb_X'] < 2050:
                        return 5
                    # 3rd column
                    if row['bomb_X'] >= 2050:
                        return 6
                # 3rd row
                if row['bomb_Y'] < 325: 
                    # 1st column
                    if row['bomb_X'] < 1900:
                        return 7
                    # 2nd column
                    if row['bomb_X'] >= 1900 and row['bomb_X'] < 2050:
                        return 8
                    # 3rd column
                    if row['bomb_X'] >= 2050:
                        return 9
        
        # If bomb is planted on B
        if row['is_bomb_planted_at_B_site'] == 1:
                # 1st row
                if row['bomb_Y'] >= 2900:
                    # 1st column
                    if row['bomb_X'] < 275:
                        return 1
                    # 2nd column
                    if row['bomb_X'] >= 275 and row['bomb_X'] < 400:
                        return 2
                    # 3rd column
                    if row['bomb_X'] >= 400:
                        return 3
                # 2nd row
                if row['bomb_Y'] < 2900 and row['bomb_Y'] >= 2725: 
                    # 1st column
                    if row['bomb_X'] < 275:
                        return 4
                    # 2nd column
                    if row['bomb_X'] >= 275 and row['bomb_X'] < 400:
                        return 5
                    # 3rd column
                    if row['bomb_X'] >= 400:
                        return 6
                # 3rd row
                if row['bomb_Y'] < 2725: 
                    # 1st column
                    if row['bomb_X'] < 275:
                        return 7
                    # 2nd column
                    if row['bomb_X'] >= 275 and row['bomb_X'] < 400:
                        return 8
                    # 3rd column
                    if row['bomb_X'] >= 400:
                        return 9

    def _TABULAR_INFERNO_bombsite_3x3_split(self, df):
            
        new_columns = pd.DataFrame({
            'bomb_mx_pos': 0
        }, index=df.index)

        df = pd.concat([df, new_columns], axis=1)
        
        df.loc[(df['is_bomb_planted_at_A_site'] == 1) | (df['is_bomb_planted_at_B_site'] == 1), 'bomb_mx_pos'] = df.apply(self.__EXT_INFERNO_get_bomb_mx_coordinate__, axis=1)

        # Dummify the bomb_mx_pos column and drop the original column
        # Poor performance
        # df['bomb_mx_pos1'] = 0
        # df['bomb_mx_pos2'] = 0
        # df['bomb_mx_pos3'] = 0
        # df['bomb_mx_pos4'] = 0
        # df['bomb_mx_pos5'] = 0
        # df['bomb_mx_pos6'] = 0
        # df['bomb_mx_pos7'] = 0
        # df['bomb_mx_pos8'] = 0
        # df['bomb_mx_pos9'] = 0

        new_columns = pd.DataFrame({
            'bomb_mx_pos1': 0,
            'bomb_mx_pos2': 0,
            'bomb_mx_pos3': 0,
            'bomb_mx_pos4': 0,
            'bomb_mx_pos5': 0,
            'bomb_mx_pos6': 0,
            'bomb_mx_pos7': 0,
            'bomb_mx_pos8': 0,
            'bomb_mx_pos9': 0
        }, index=df.index)

        df = pd.concat([df, new_columns], axis=1)

        df.loc[df['bomb_mx_pos'] == 1, 'bomb_mx_pos1'] = 1
        df.loc[df['bomb_mx_pos'] == 2, 'bomb_mx_pos2'] = 1
        df.loc[df['bomb_mx_pos'] == 3, 'bomb_mx_pos3'] = 1
        df.loc[df['bomb_mx_pos'] == 4, 'bomb_mx_pos4'] = 1
        df.loc[df['bomb_mx_pos'] == 5, 'bomb_mx_pos5'] = 1
        df.loc[df['bomb_mx_pos'] == 6, 'bomb_mx_pos6'] = 1
        df.loc[df['bomb_mx_pos'] == 7, 'bomb_mx_pos7'] = 1
        df.loc[df['bomb_mx_pos'] == 8, 'bomb_mx_pos8'] = 1
        df.loc[df['bomb_mx_pos'] == 9, 'bomb_mx_pos9'] = 1

        df = df.drop(columns=['bomb_mx_pos'])

        return df
    


    # 10. Handle smoke and molotov grenades
    def _TABULAR_smokes_and_molotovs(self, df, smokes, infernos):

        # Create new columns for smokes and infernos in the tabular dataframe
        new_columns = pd.DataFrame({
            'smokes_active': [[] for _ in range(len(df))],
            'infernos_active': [[] for _ in range(len(df))]
        }, index=df.index)

        df = pd.concat([df, new_columns], axis=1)

        # Handle smokes
        for _, row in smokes.iterrows():

            startTick = row['start_tick']
            endTick = row['end_tick']
            df.loc[(df['round'] == row['round']) & (df['tick'] >= startTick) & (df['tick'] <= endTick), 'smokes_active'].apply(lambda x: x.append([row['X'], row['Y'], row['Z']]))
        
        for _, row in infernos.iterrows():

            startTick = row['start_tick']
            endTick = row['end_tick']
            df.loc[(df['round'] == row['round']) & (df['tick'] >= startTick) & (df['tick'] <= endTick), 'infernos_active'].apply(lambda x: x.append([row['X'], row['Y'], row['Z']]))

        return df



    # 11. Add numerical match id
    def _TABULAR_numerical_match_id(self, tabular_df):

        if type(self.numerical_match_id) is not int:
            raise ValueError("Numerical match id must be an integer.")
        
        new_columns = pd.DataFrame({
            'numerical_match_id': self.numerical_match_id
        }, index=tabular_df.index)
        tabular_df = pd.concat([tabular_df, new_columns], axis=1)

        return tabular_df



    # 12. Function to extend the dataframe with copies of the rounds with varied player permutations
    def _TABULAR_player_permutation(self, df, num_permutations_per_round=3):
        """
        Function to extend the dataframe with copies of the rounds with varied player permutations
        """

        # Get the unique rounds and store team 1 and two player numbers
        team_1_indicies = [0, 1, 2, 3, 4]
        team_2_indicies = [5, 6, 7, 8, 9]
        rounds = df['round'].unique()

        for rnd in rounds:
            for _permutation in range(num_permutations_per_round):
                # Get the round dataframe
                round_df = df[df['round'] == rnd].copy()
                round_df = round_df.reset_index(drop=True)

                # Rename all columns starting with 'player' to start with 'playerPERM'
                player_cols = [col for col in round_df.columns if col.startswith('player')]
                for col in player_cols:
                    round_df.rename(columns={col: 'playerPERM' + col[6:]}, inplace=True)

                # Get random permutations for both teams
                random.shuffle(team_1_indicies)
                random.shuffle(team_1_indicies)

                # Player columns
                player_0_cols = [col for col in round_df.columns if col.startswith('playerPERM0')]
                player_1_cols = [col for col in round_df.columns if col.startswith('playerPERM1')]
                player_2_cols = [col for col in round_df.columns if col.startswith('playerPERM2')]
                player_3_cols = [col for col in round_df.columns if col.startswith('playerPERM3')]
                player_4_cols = [col for col in round_df.columns if col.startswith('playerPERM4')]

                player_5_cols = [col for col in round_df.columns if col.startswith('playerPERM5')]
                player_6_cols = [col for col in round_df.columns if col.startswith('playerPERM6')]
                player_7_cols = [col for col in round_df.columns if col.startswith('playerPERM7')]
                player_8_cols = [col for col in round_df.columns if col.startswith('playerPERM8')]
                player_9_cols = [col for col in round_df.columns if col.startswith('playerPERM9')]

                # Rewrite the player columns with the new permutations
                for col in player_0_cols:
                    round_df.rename(columns={col: 'player' + str(team_1_indicies[0]) + col[11:]}, inplace=True)
                for col in player_1_cols:
                    round_df.rename(columns={col: 'player' + str(team_1_indicies[1]) + col[11:]}, inplace=True)
                for col in player_2_cols:
                    round_df.rename(columns={col: 'player' + str(team_1_indicies[2]) + col[11:]}, inplace=True)
                for col in player_3_cols:
                    round_df.rename(columns={col: 'player' + str(team_1_indicies[3]) + col[11:]}, inplace=True)
                for col in player_4_cols:
                    round_df.rename(columns={col: 'player' + str(team_1_indicies[4]) + col[11:]}, inplace=True)

                for col in player_5_cols:
                    round_df.rename(columns={col: 'player' + str(team_2_indicies[0]) + col[11:]}, inplace=True)
                for col in player_6_cols:
                    round_df.rename(columns={col: 'player' + str(team_2_indicies[1]) + col[11:]}, inplace=True)
                for col in player_7_cols:
                    round_df.rename(columns={col: 'player' + str(team_2_indicies[2]) + col[11:]}, inplace=True)
                for col in player_8_cols:
                    round_df.rename(columns={col: 'player' + str(team_2_indicies[3]) + col[11:]}, inplace=True)
                for col in player_9_cols:
                    round_df.rename(columns={col: 'player' + str(team_2_indicies[4]) + col[11:]}, inplace=True)

                # Append the new round to the dataframe
                df = pd.concat([df, round_df])

        return df



    # 13. Rearrange the player columns so that the CTs are always from 0 to 4 and Ts are from 5 to 9
    def _TABULAR_refactor_player_columns(self, df):

        # Separate the CT and T halves
        team_1_ct = df.loc[df['player0_is_CT'] == True].copy()
        team_2_ct = df.loc[df['player0_is_CT'] == False].copy()

        # Rename the columns for team_1_ct
        for col in team_1_ct.columns:
            if col.startswith('player') and int(col[6]) <= 4:
                team_1_ct.rename(columns={col: col.replace('player', 'CT')}, inplace=True)
            elif col.startswith('player') and int(col[6]) > 4:
                team_1_ct.rename(columns={col: col.replace('player', 'T')}, inplace=True)

        # Rename the columns for team_2_ct
        for col in team_2_ct.columns:
            if col.startswith('player') and int(col[6]) <= 4:
                team_2_ct.rename(columns={col: col.replace('player' + col[6],  'T' + str(int(col[6]) + 5))}, inplace=True)
            elif col.startswith('player') and int(col[6]) > 4:
                team_2_ct.rename(columns={col: col.replace('player' + col[6], 'CT' + str(int(col[6]) - 5))}, inplace=True)

        # Column order
        col_order = [
            'CT0_name', 'CT0_X', 'CT0_Y', 'CT0_Z', 'CT0_pitch', 'CT0_yaw', 'CT0_velocity_X', 'CT0_velocity_Y', 'CT0_velocity_Z', 'CT0_health', 'CT0_armor_value', 'CT0_active_weapon_ammo', 'CT0_total_ammo_left', 'CT0_flash_duration', 'CT0_balance', 'CT0_current_equip_value', 'CT0_round_start_equip_value', 
            'CT0_is_alive', 'CT0_is_CT', 'CT0_is_shooting', 'CT0_is_crouching', 'CT0_is_ducking', 'CT0_is_duck_jumping', 'CT0_is_walking', 'CT0_is_spotted', 'CT0_is_scoped', 'CT0_is_defusing', 'CT0_is_reloading', 'CT0_is_in_bombsite',
            'CT0_stat_kills', 'CT0_stat_HS_kills', 'CT0_stat_opening_kills', 'CT0_stat_deaths', 'CT0_stat_opening_deaths', 'CT0_stat_assists', 'CT0_stat_flash_assists', 'CT0_stat_damage', 'CT0_stat_weapon_damage', 'CT0_stat_nade_damage', 'CT0_stat_survives', 'CT0_stat_KPR', 'CT0_stat_ADR', 'CT0_stat_DPR', 'CT0_stat_HS%', 'CT0_stat_SPR', 
            'CT0_inventory_C4', 'CT0_inventory_Taser', 'CT0_inventory_USP-S', 'CT0_inventory_P2000', 'CT0_inventory_Glock-18', 'CT0_inventory_Dual Berettas', 'CT0_inventory_P250', 'CT0_inventory_Tec-9', 'CT0_inventory_CZ75 Auto', 'CT0_inventory_Five-SeveN', 'CT0_inventory_Desert Eagle', 'CT0_inventory_MAC-10', 'CT0_inventory_MP9', 'CT0_inventory_MP7', 'CT0_inventory_MP5-SD', 'CT0_inventory_UMP-45', 'CT0_inventory_PP-Bizon', 'CT0_inventory_P90', 'CT0_inventory_Nova', 'CT0_inventory_XM1014', 'CT0_inventory_Sawed-Off', 'CT0_inventory_MAG-7', 'CT0_inventory_M249', 'CT0_inventory_Negev', 'CT0_inventory_FAMAS', 'CT0_inventory_Galil AR', 'CT0_inventory_AK-47', 'CT0_inventory_M4A4', 'CT0_inventory_M4A1-S', 'CT0_inventory_SG 553', 'CT0_inventory_AUG', 'CT0_inventory_SSG 08', 'CT0_inventory_AWP', 'CT0_inventory_G3SG1', 'CT0_inventory_SCAR-20', 'CT0_inventory_HE Grenade', 'CT0_inventory_Flashbang', 'CT0_inventory_Smoke Grenade', 'CT0_inventory_Incendiary Grenade', 'CT0_inventory_Molotov', 'CT0_inventory_Decoy Grenade',
            'CT0_active_weapon_C4', 'CT0_active_weapon_Knife', 'CT0_active_weapon_Taser', 'CT0_active_weapon_USP-S', 'CT0_active_weapon_P2000', 'CT0_active_weapon_Glock-18', 'CT0_active_weapon_Dual Berettas', 'CT0_active_weapon_P250', 'CT0_active_weapon_Tec-9', 'CT0_active_weapon_CZ75 Auto', 'CT0_active_weapon_Five-SeveN', 'CT0_active_weapon_Desert Eagle', 'CT0_active_weapon_MAC-10', 'CT0_active_weapon_MP9', 'CT0_active_weapon_MP7', 'CT0_active_weapon_MP5-SD', 'CT0_active_weapon_UMP-45', 'CT0_active_weapon_PP-Bizon', 'CT0_active_weapon_P90', 'CT0_active_weapon_Nova', 'CT0_active_weapon_XM1014', 'CT0_active_weapon_Sawed-Off', 'CT0_active_weapon_MAG-7', 'CT0_active_weapon_M249', 'CT0_active_weapon_Negev', 'CT0_active_weapon_FAMAS', 'CT0_active_weapon_Galil AR', 'CT0_active_weapon_AK-47', 'CT0_active_weapon_M4A4', 'CT0_active_weapon_M4A1-S', 'CT0_active_weapon_SG 553', 'CT0_active_weapon_AUG', 'CT0_active_weapon_SSG 08', 'CT0_active_weapon_AWP', 'CT0_active_weapon_G3SG1', 'CT0_active_weapon_SCAR-20', 'CT0_active_weapon_HE Grenade', 'CT0_active_weapon_Flashbang', 'CT0_active_weapon_Smoke Grenade', 'CT0_active_weapon_Incendiary Grenade', 'CT0_active_weapon_Molotov', 'CT0_active_weapon_Decoy Grenade',
            'CT0_hltv_rating_2.0', 'CT0_hltv_DPR', 'CT0_hltv_KAST', 'CT0_hltv_Impact', 'CT0_hltv_ADR', 'CT0_hltv_KPR', 'CT0_hltv_total_kills', 'CT0_hltv_HS%', 'CT0_hltv_total_deaths', 'CT0_hltv_KD_ratio', 'CT0_hltv_dmgPR', 'CT0_hltv_grenade_dmgPR', 'CT0_hltv_maps_played', 'CT0_hltv_saved_by_teammatePR', 'CT0_hltv_saved_teammatesPR', 'CT0_hltv_opening_kill_rating', 'CT0_hltv_team_W%_after_opening', 'CT0_hltv_opening_kill_in_W_rounds', 'CT0_hltv_rating_1.0_all_Career', 'CT0_hltv_clutches_1on1_ratio', 'CT0_hltv_clutches_won_1on1', 'CT0_hltv_clutches_won_1on2', 'CT0_hltv_clutches_won_1on3', 'CT0_hltv_clutches_won_1on4', 'CT0_hltv_clutches_won_1on5', 
                
            'CT1_name', 'CT1_X', 'CT1_Y', 'CT1_Z', 'CT1_pitch', 'CT1_yaw', 'CT1_velocity_X', 'CT1_velocity_Y', 'CT1_velocity_Z', 'CT1_health', 'CT1_armor_value', 'CT1_active_weapon_ammo', 'CT1_total_ammo_left', 'CT1_flash_duration', 'CT1_balance', 'CT1_current_equip_value', 'CT1_round_start_equip_value', 
            'CT1_is_alive', 'CT1_is_CT', 'CT1_is_shooting', 'CT1_is_crouching', 'CT1_is_ducking', 'CT1_is_duck_jumping', 'CT1_is_walking', 'CT1_is_spotted', 'CT1_is_scoped', 'CT1_is_defusing', 'CT1_is_reloading', 'CT1_is_in_bombsite',
            'CT1_stat_kills', 'CT1_stat_HS_kills', 'CT1_stat_opening_kills', 'CT1_stat_deaths', 'CT1_stat_opening_deaths', 'CT1_stat_assists', 'CT1_stat_flash_assists', 'CT1_stat_damage', 'CT1_stat_weapon_damage', 'CT1_stat_nade_damage', 'CT1_stat_survives', 'CT1_stat_KPR', 'CT1_stat_ADR', 'CT1_stat_DPR', 'CT1_stat_HS%', 'CT1_stat_SPR', 
            'CT1_inventory_C4', 'CT1_inventory_Taser', 'CT1_inventory_USP-S', 'CT1_inventory_P2000', 'CT1_inventory_Glock-18', 'CT1_inventory_Dual Berettas', 'CT1_inventory_P250', 'CT1_inventory_Tec-9', 'CT1_inventory_CZ75 Auto', 'CT1_inventory_Five-SeveN', 'CT1_inventory_Desert Eagle', 'CT1_inventory_MAC-10', 'CT1_inventory_MP9', 'CT1_inventory_MP7', 'CT1_inventory_MP5-SD', 'CT1_inventory_UMP-45', 'CT1_inventory_PP-Bizon', 'CT1_inventory_P90', 'CT1_inventory_Nova', 'CT1_inventory_XM1014', 'CT1_inventory_Sawed-Off', 'CT1_inventory_MAG-7', 'CT1_inventory_M249', 'CT1_inventory_Negev', 'CT1_inventory_FAMAS', 'CT1_inventory_Galil AR', 'CT1_inventory_AK-47', 'CT1_inventory_M4A4', 'CT1_inventory_M4A1-S', 'CT1_inventory_SG 553', 'CT1_inventory_AUG', 'CT1_inventory_SSG 08', 'CT1_inventory_AWP', 'CT1_inventory_G3SG1', 'CT1_inventory_SCAR-20', 'CT1_inventory_HE Grenade', 'CT1_inventory_Flashbang', 'CT1_inventory_Smoke Grenade', 'CT1_inventory_Incendiary Grenade', 'CT1_inventory_Molotov', 'CT1_inventory_Decoy Grenade',
            'CT1_active_weapon_C4', 'CT1_active_weapon_Knife', 'CT1_active_weapon_Taser', 'CT1_active_weapon_USP-S', 'CT1_active_weapon_P2000', 'CT1_active_weapon_Glock-18', 'CT1_active_weapon_Dual Berettas', 'CT1_active_weapon_P250', 'CT1_active_weapon_Tec-9', 'CT1_active_weapon_CZ75 Auto', 'CT1_active_weapon_Five-SeveN', 'CT1_active_weapon_Desert Eagle', 'CT1_active_weapon_MAC-10', 'CT1_active_weapon_MP9', 'CT1_active_weapon_MP7', 'CT1_active_weapon_MP5-SD', 'CT1_active_weapon_UMP-45', 'CT1_active_weapon_PP-Bizon', 'CT1_active_weapon_P90', 'CT1_active_weapon_Nova', 'CT1_active_weapon_XM1014', 'CT1_active_weapon_Sawed-Off', 'CT1_active_weapon_MAG-7', 'CT1_active_weapon_M249', 'CT1_active_weapon_Negev', 'CT1_active_weapon_FAMAS', 'CT1_active_weapon_Galil AR', 'CT1_active_weapon_AK-47', 'CT1_active_weapon_M4A4', 'CT1_active_weapon_M4A1-S', 'CT1_active_weapon_SG 553', 'CT1_active_weapon_AUG', 'CT1_active_weapon_SSG 08', 'CT1_active_weapon_AWP', 'CT1_active_weapon_G3SG1', 'CT1_active_weapon_SCAR-20', 'CT1_active_weapon_HE Grenade', 'CT1_active_weapon_Flashbang', 'CT1_active_weapon_Smoke Grenade', 'CT1_active_weapon_Incendiary Grenade', 'CT1_active_weapon_Molotov', 'CT1_active_weapon_Decoy Grenade',
            'CT1_hltv_rating_2.0', 'CT1_hltv_DPR', 'CT1_hltv_KAST', 'CT1_hltv_Impact', 'CT1_hltv_ADR', 'CT1_hltv_KPR', 'CT1_hltv_total_kills', 'CT1_hltv_HS%', 'CT1_hltv_total_deaths', 'CT1_hltv_KD_ratio', 'CT1_hltv_dmgPR', 'CT1_hltv_grenade_dmgPR', 'CT1_hltv_maps_played', 'CT1_hltv_saved_by_teammatePR', 'CT1_hltv_saved_teammatesPR', 'CT1_hltv_opening_kill_rating', 'CT1_hltv_team_W%_after_opening', 'CT1_hltv_opening_kill_in_W_rounds', 'CT1_hltv_rating_1.0_all_Career', 'CT1_hltv_clutches_1on1_ratio', 'CT1_hltv_clutches_won_1on1', 'CT1_hltv_clutches_won_1on2', 'CT1_hltv_clutches_won_1on3', 'CT1_hltv_clutches_won_1on4', 'CT1_hltv_clutches_won_1on5', 

            'CT2_name', 'CT2_X', 'CT2_Y', 'CT2_Z', 'CT2_pitch', 'CT2_yaw', 'CT2_velocity_X', 'CT2_velocity_Y', 'CT2_velocity_Z', 'CT2_health', 'CT2_armor_value', 'CT2_active_weapon_ammo', 'CT2_total_ammo_left', 'CT2_flash_duration', 'CT2_balance', 'CT2_current_equip_value', 'CT2_round_start_equip_value', 
            'CT2_is_alive', 'CT2_is_CT', 'CT2_is_shooting', 'CT2_is_crouching', 'CT2_is_ducking', 'CT2_is_duck_jumping', 'CT2_is_walking', 'CT2_is_spotted', 'CT2_is_scoped', 'CT2_is_defusing', 'CT2_is_reloading', 'CT2_is_in_bombsite',
            'CT2_stat_kills', 'CT2_stat_HS_kills', 'CT2_stat_opening_kills', 'CT2_stat_deaths', 'CT2_stat_opening_deaths', 'CT2_stat_assists', 'CT2_stat_flash_assists', 'CT2_stat_damage', 'CT2_stat_weapon_damage', 'CT2_stat_nade_damage', 'CT2_stat_survives', 'CT2_stat_KPR', 'CT2_stat_ADR', 'CT2_stat_DPR', 'CT2_stat_HS%', 'CT2_stat_SPR', 
            'CT2_inventory_C4', 'CT2_inventory_Taser', 'CT2_inventory_USP-S', 'CT2_inventory_P2000', 'CT2_inventory_Glock-18', 'CT2_inventory_Dual Berettas', 'CT2_inventory_P250', 'CT2_inventory_Tec-9', 'CT2_inventory_CZ75 Auto', 'CT2_inventory_Five-SeveN', 'CT2_inventory_Desert Eagle', 'CT2_inventory_MAC-10', 'CT2_inventory_MP9', 'CT2_inventory_MP7', 'CT2_inventory_MP5-SD', 'CT2_inventory_UMP-45', 'CT2_inventory_PP-Bizon', 'CT2_inventory_P90', 'CT2_inventory_Nova', 'CT2_inventory_XM1014', 'CT2_inventory_Sawed-Off', 'CT2_inventory_MAG-7', 'CT2_inventory_M249', 'CT2_inventory_Negev', 'CT2_inventory_FAMAS', 'CT2_inventory_Galil AR', 'CT2_inventory_AK-47', 'CT2_inventory_M4A4', 'CT2_inventory_M4A1-S', 'CT2_inventory_SG 553', 'CT2_inventory_AUG', 'CT2_inventory_SSG 08', 'CT2_inventory_AWP', 'CT2_inventory_G3SG1', 'CT2_inventory_SCAR-20', 'CT2_inventory_HE Grenade', 'CT2_inventory_Flashbang', 'CT2_inventory_Smoke Grenade', 'CT2_inventory_Incendiary Grenade', 'CT2_inventory_Molotov', 'CT2_inventory_Decoy Grenade',
            'CT2_active_weapon_C4', 'CT2_active_weapon_Knife', 'CT2_active_weapon_Taser', 'CT2_active_weapon_USP-S', 'CT2_active_weapon_P2000', 'CT2_active_weapon_Glock-18', 'CT2_active_weapon_Dual Berettas', 'CT2_active_weapon_P250', 'CT2_active_weapon_Tec-9', 'CT2_active_weapon_CZ75 Auto', 'CT2_active_weapon_Five-SeveN', 'CT2_active_weapon_Desert Eagle', 'CT2_active_weapon_MAC-10', 'CT2_active_weapon_MP9', 'CT2_active_weapon_MP7', 'CT2_active_weapon_MP5-SD', 'CT2_active_weapon_UMP-45', 'CT2_active_weapon_PP-Bizon', 'CT2_active_weapon_P90', 'CT2_active_weapon_Nova', 'CT2_active_weapon_XM1014', 'CT2_active_weapon_Sawed-Off', 'CT2_active_weapon_MAG-7', 'CT2_active_weapon_M249', 'CT2_active_weapon_Negev', 'CT2_active_weapon_FAMAS', 'CT2_active_weapon_Galil AR', 'CT2_active_weapon_AK-47', 'CT2_active_weapon_M4A4', 'CT2_active_weapon_M4A1-S', 'CT2_active_weapon_SG 553', 'CT2_active_weapon_AUG', 'CT2_active_weapon_SSG 08', 'CT2_active_weapon_AWP', 'CT2_active_weapon_G3SG1', 'CT2_active_weapon_SCAR-20', 'CT2_active_weapon_HE Grenade', 'CT2_active_weapon_Flashbang', 'CT2_active_weapon_Smoke Grenade', 'CT2_active_weapon_Incendiary Grenade', 'CT2_active_weapon_Molotov', 'CT2_active_weapon_Decoy Grenade',
            'CT2_hltv_rating_2.0', 'CT2_hltv_DPR', 'CT2_hltv_KAST', 'CT2_hltv_Impact', 'CT2_hltv_ADR', 'CT2_hltv_KPR', 'CT2_hltv_total_kills', 'CT2_hltv_HS%', 'CT2_hltv_total_deaths', 'CT2_hltv_KD_ratio', 'CT2_hltv_dmgPR', 'CT2_hltv_grenade_dmgPR', 'CT2_hltv_maps_played', 'CT2_hltv_saved_by_teammatePR', 'CT2_hltv_saved_teammatesPR', 'CT2_hltv_opening_kill_rating', 'CT2_hltv_team_W%_after_opening', 'CT2_hltv_opening_kill_in_W_rounds', 'CT2_hltv_rating_1.0_all_Career', 'CT2_hltv_clutches_1on1_ratio', 'CT2_hltv_clutches_won_1on1', 'CT2_hltv_clutches_won_1on2', 'CT2_hltv_clutches_won_1on3', 'CT2_hltv_clutches_won_1on4', 'CT2_hltv_clutches_won_1on5', 

            'CT3_name', 'CT3_X', 'CT3_Y', 'CT3_Z', 'CT3_pitch', 'CT3_yaw', 'CT3_velocity_X', 'CT3_velocity_Y', 'CT3_velocity_Z', 'CT3_health', 'CT3_armor_value', 'CT3_active_weapon_ammo', 'CT3_total_ammo_left', 'CT3_flash_duration', 'CT3_balance', 'CT3_current_equip_value', 'CT3_round_start_equip_value', 
            'CT3_is_alive', 'CT3_is_CT', 'CT3_is_shooting', 'CT3_is_crouching', 'CT3_is_ducking', 'CT3_is_duck_jumping', 'CT3_is_walking', 'CT3_is_spotted', 'CT3_is_scoped', 'CT3_is_defusing', 'CT3_is_reloading', 'CT3_is_in_bombsite',
            'CT3_stat_kills', 'CT3_stat_HS_kills', 'CT3_stat_opening_kills', 'CT3_stat_deaths', 'CT3_stat_opening_deaths', 'CT3_stat_assists', 'CT3_stat_flash_assists', 'CT3_stat_damage', 'CT3_stat_weapon_damage', 'CT3_stat_nade_damage', 'CT3_stat_survives', 'CT3_stat_KPR', 'CT3_stat_ADR', 'CT3_stat_DPR', 'CT3_stat_HS%', 'CT3_stat_SPR', 
            'CT3_inventory_C4', 'CT3_inventory_Taser', 'CT3_inventory_USP-S', 'CT3_inventory_P2000', 'CT3_inventory_Glock-18', 'CT3_inventory_Dual Berettas', 'CT3_inventory_P250', 'CT3_inventory_Tec-9', 'CT3_inventory_CZ75 Auto', 'CT3_inventory_Five-SeveN', 'CT3_inventory_Desert Eagle', 'CT3_inventory_MAC-10', 'CT3_inventory_MP9', 'CT3_inventory_MP7', 'CT3_inventory_MP5-SD', 'CT3_inventory_UMP-45', 'CT3_inventory_PP-Bizon', 'CT3_inventory_P90', 'CT3_inventory_Nova', 'CT3_inventory_XM1014', 'CT3_inventory_Sawed-Off', 'CT3_inventory_MAG-7', 'CT3_inventory_M249', 'CT3_inventory_Negev', 'CT3_inventory_FAMAS', 'CT3_inventory_Galil AR', 'CT3_inventory_AK-47', 'CT3_inventory_M4A4', 'CT3_inventory_M4A1-S', 'CT3_inventory_SG 553', 'CT3_inventory_AUG', 'CT3_inventory_SSG 08', 'CT3_inventory_AWP', 'CT3_inventory_G3SG1', 'CT3_inventory_SCAR-20', 'CT3_inventory_HE Grenade', 'CT3_inventory_Flashbang', 'CT3_inventory_Smoke Grenade', 'CT3_inventory_Incendiary Grenade', 'CT3_inventory_Molotov', 'CT3_inventory_Decoy Grenade',
            'CT3_active_weapon_C4', 'CT3_active_weapon_Knife', 'CT3_active_weapon_Taser', 'CT3_active_weapon_USP-S', 'CT3_active_weapon_P2000', 'CT3_active_weapon_Glock-18', 'CT3_active_weapon_Dual Berettas', 'CT3_active_weapon_P250', 'CT3_active_weapon_Tec-9', 'CT3_active_weapon_CZ75 Auto', 'CT3_active_weapon_Five-SeveN', 'CT3_active_weapon_Desert Eagle', 'CT3_active_weapon_MAC-10', 'CT3_active_weapon_MP9', 'CT3_active_weapon_MP7', 'CT3_active_weapon_MP5-SD', 'CT3_active_weapon_UMP-45', 'CT3_active_weapon_PP-Bizon', 'CT3_active_weapon_P90', 'CT3_active_weapon_Nova', 'CT3_active_weapon_XM1014', 'CT3_active_weapon_Sawed-Off', 'CT3_active_weapon_MAG-7', 'CT3_active_weapon_M249', 'CT3_active_weapon_Negev', 'CT3_active_weapon_FAMAS', 'CT3_active_weapon_Galil AR', 'CT3_active_weapon_AK-47', 'CT3_active_weapon_M4A4', 'CT3_active_weapon_M4A1-S', 'CT3_active_weapon_SG 553', 'CT3_active_weapon_AUG', 'CT3_active_weapon_SSG 08', 'CT3_active_weapon_AWP', 'CT3_active_weapon_G3SG1', 'CT3_active_weapon_SCAR-20', 'CT3_active_weapon_HE Grenade', 'CT3_active_weapon_Flashbang', 'CT3_active_weapon_Smoke Grenade', 'CT3_active_weapon_Incendiary Grenade', 'CT3_active_weapon_Molotov', 'CT3_active_weapon_Decoy Grenade',
            'CT3_hltv_rating_2.0', 'CT3_hltv_DPR', 'CT3_hltv_KAST', 'CT3_hltv_Impact', 'CT3_hltv_ADR', 'CT3_hltv_KPR', 'CT3_hltv_total_kills', 'CT3_hltv_HS%', 'CT3_hltv_total_deaths', 'CT3_hltv_KD_ratio', 'CT3_hltv_dmgPR', 'CT3_hltv_grenade_dmgPR', 'CT3_hltv_maps_played', 'CT3_hltv_saved_by_teammatePR', 'CT3_hltv_saved_teammatesPR', 'CT3_hltv_opening_kill_rating', 'CT3_hltv_team_W%_after_opening', 'CT3_hltv_opening_kill_in_W_rounds', 'CT3_hltv_rating_1.0_all_Career', 'CT3_hltv_clutches_1on1_ratio', 'CT3_hltv_clutches_won_1on1', 'CT3_hltv_clutches_won_1on2', 'CT3_hltv_clutches_won_1on3', 'CT3_hltv_clutches_won_1on4', 'CT3_hltv_clutches_won_1on5', 

            'CT4_name', 'CT4_X', 'CT4_Y', 'CT4_Z', 'CT4_pitch', 'CT4_yaw', 'CT4_velocity_X', 'CT4_velocity_Y', 'CT4_velocity_Z', 'CT4_health', 'CT4_armor_value', 'CT4_active_weapon_ammo', 'CT4_total_ammo_left', 'CT4_flash_duration', 'CT4_balance', 'CT4_current_equip_value', 'CT4_round_start_equip_value', 
            'CT4_is_alive', 'CT4_is_CT', 'CT4_is_shooting', 'CT4_is_crouching', 'CT4_is_ducking', 'CT4_is_duck_jumping', 'CT4_is_walking', 'CT4_is_spotted', 'CT4_is_scoped', 'CT4_is_defusing', 'CT4_is_reloading', 'CT4_is_in_bombsite',
            'CT4_stat_kills', 'CT4_stat_HS_kills', 'CT4_stat_opening_kills', 'CT4_stat_deaths', 'CT4_stat_opening_deaths', 'CT4_stat_assists', 'CT4_stat_flash_assists', 'CT4_stat_damage', 'CT4_stat_weapon_damage', 'CT4_stat_nade_damage', 'CT4_stat_survives', 'CT4_stat_KPR', 'CT4_stat_ADR', 'CT4_stat_DPR', 'CT4_stat_HS%', 'CT4_stat_SPR', 
            'CT4_inventory_C4', 'CT4_inventory_Taser', 'CT4_inventory_USP-S', 'CT4_inventory_P2000', 'CT4_inventory_Glock-18', 'CT4_inventory_Dual Berettas', 'CT4_inventory_P250', 'CT4_inventory_Tec-9', 'CT4_inventory_CZ75 Auto', 'CT4_inventory_Five-SeveN', 'CT4_inventory_Desert Eagle', 'CT4_inventory_MAC-10', 'CT4_inventory_MP9', 'CT4_inventory_MP7', 'CT4_inventory_MP5-SD', 'CT4_inventory_UMP-45', 'CT4_inventory_PP-Bizon', 'CT4_inventory_P90', 'CT4_inventory_Nova', 'CT4_inventory_XM1014', 'CT4_inventory_Sawed-Off', 'CT4_inventory_MAG-7', 'CT4_inventory_M249', 'CT4_inventory_Negev', 'CT4_inventory_FAMAS', 'CT4_inventory_Galil AR', 'CT4_inventory_AK-47', 'CT4_inventory_M4A4', 'CT4_inventory_M4A1-S', 'CT4_inventory_SG 553', 'CT4_inventory_AUG', 'CT4_inventory_SSG 08', 'CT4_inventory_AWP', 'CT4_inventory_G3SG1', 'CT4_inventory_SCAR-20', 'CT4_inventory_HE Grenade', 'CT4_inventory_Flashbang', 'CT4_inventory_Smoke Grenade', 'CT4_inventory_Incendiary Grenade', 'CT4_inventory_Molotov', 'CT4_inventory_Decoy Grenade',
            'CT4_active_weapon_C4', 'CT4_active_weapon_Knife', 'CT4_active_weapon_Taser', 'CT4_active_weapon_USP-S', 'CT4_active_weapon_P2000', 'CT4_active_weapon_Glock-18', 'CT4_active_weapon_Dual Berettas', 'CT4_active_weapon_P250', 'CT4_active_weapon_Tec-9', 'CT4_active_weapon_CZ75 Auto', 'CT4_active_weapon_Five-SeveN', 'CT4_active_weapon_Desert Eagle', 'CT4_active_weapon_MAC-10', 'CT4_active_weapon_MP9', 'CT4_active_weapon_MP7', 'CT4_active_weapon_MP5-SD', 'CT4_active_weapon_UMP-45', 'CT4_active_weapon_PP-Bizon', 'CT4_active_weapon_P90', 'CT4_active_weapon_Nova', 'CT4_active_weapon_XM1014', 'CT4_active_weapon_Sawed-Off', 'CT4_active_weapon_MAG-7', 'CT4_active_weapon_M249', 'CT4_active_weapon_Negev', 'CT4_active_weapon_FAMAS', 'CT4_active_weapon_Galil AR', 'CT4_active_weapon_AK-47', 'CT4_active_weapon_M4A4', 'CT4_active_weapon_M4A1-S', 'CT4_active_weapon_SG 553', 'CT4_active_weapon_AUG', 'CT4_active_weapon_SSG 08', 'CT4_active_weapon_AWP', 'CT4_active_weapon_G3SG1', 'CT4_active_weapon_SCAR-20', 'CT4_active_weapon_HE Grenade', 'CT4_active_weapon_Flashbang', 'CT4_active_weapon_Smoke Grenade', 'CT4_active_weapon_Incendiary Grenade', 'CT4_active_weapon_Molotov', 'CT4_active_weapon_Decoy Grenade',
            'CT4_hltv_rating_2.0', 'CT4_hltv_DPR', 'CT4_hltv_KAST', 'CT4_hltv_Impact', 'CT4_hltv_ADR', 'CT4_hltv_KPR', 'CT4_hltv_total_kills', 'CT4_hltv_HS%', 'CT4_hltv_total_deaths', 'CT4_hltv_KD_ratio', 'CT4_hltv_dmgPR', 'CT4_hltv_grenade_dmgPR', 'CT4_hltv_maps_played', 'CT4_hltv_saved_by_teammatePR', 'CT4_hltv_saved_teammatesPR', 'CT4_hltv_opening_kill_rating', 'CT4_hltv_team_W%_after_opening', 'CT4_hltv_opening_kill_in_W_rounds', 'CT4_hltv_rating_1.0_all_Career', 'CT4_hltv_clutches_1on1_ratio', 'CT4_hltv_clutches_won_1on1', 'CT4_hltv_clutches_won_1on2', 'CT4_hltv_clutches_won_1on3', 'CT4_hltv_clutches_won_1on4', 'CT4_hltv_clutches_won_1on5', 

            'T5_name', 'T5_X', 'T5_Y', 'T5_Z', 'T5_pitch', 'T5_yaw', 'T5_velocity_X', 'T5_velocity_Y', 'T5_velocity_Z', 'T5_health', 'T5_armor_value', 'T5_active_weapon_ammo', 'T5_total_ammo_left', 'T5_flash_duration', 'T5_balance', 'T5_current_equip_value', 'T5_round_start_equip_value', 
            'T5_is_alive', 'T5_is_CT', 'T5_is_shooting', 'T5_is_crouching', 'T5_is_ducking', 'T5_is_duck_jumping', 'T5_is_walking', 'T5_is_spotted', 'T5_is_scoped', 'T5_is_defusing', 'T5_is_reloading', 'T5_is_in_bombsite',
            'T5_stat_kills', 'T5_stat_HS_kills', 'T5_stat_opening_kills', 'T5_stat_deaths', 'T5_stat_opening_deaths', 'T5_stat_assists', 'T5_stat_flash_assists', 'T5_stat_damage', 'T5_stat_weapon_damage', 'T5_stat_nade_damage', 'T5_stat_survives', 'T5_stat_KPR', 'T5_stat_ADR', 'T5_stat_DPR', 'T5_stat_HS%', 'T5_stat_SPR', 
            'T5_inventory_C4', 'T5_inventory_Taser', 'T5_inventory_USP-S', 'T5_inventory_P2000', 'T5_inventory_Glock-18', 'T5_inventory_Dual Berettas', 'T5_inventory_P250', 'T5_inventory_Tec-9', 'T5_inventory_CZ75 Auto', 'T5_inventory_Five-SeveN', 'T5_inventory_Desert Eagle', 'T5_inventory_MAC-10', 'T5_inventory_MP9', 'T5_inventory_MP7', 'T5_inventory_MP5-SD', 'T5_inventory_UMP-45', 'T5_inventory_PP-Bizon', 'T5_inventory_P90', 'T5_inventory_Nova', 'T5_inventory_XM1014', 'T5_inventory_Sawed-Off', 'T5_inventory_MAG-7', 'T5_inventory_M249', 'T5_inventory_Negev', 'T5_inventory_FAMAS', 'T5_inventory_Galil AR', 'T5_inventory_AK-47', 'T5_inventory_M4A4', 'T5_inventory_M4A1-S', 'T5_inventory_SG 553', 'T5_inventory_AUG', 'T5_inventory_SSG 08', 'T5_inventory_AWP', 'T5_inventory_G3SG1', 'T5_inventory_SCAR-20', 'T5_inventory_HE Grenade', 'T5_inventory_Flashbang', 'T5_inventory_Smoke Grenade', 'T5_inventory_Incendiary Grenade', 'T5_inventory_Molotov', 'T5_inventory_Decoy Grenade',
            'T5_active_weapon_C4', 'T5_active_weapon_Knife', 'T5_active_weapon_Taser', 'T5_active_weapon_USP-S', 'T5_active_weapon_P2000', 'T5_active_weapon_Glock-18', 'T5_active_weapon_Dual Berettas', 'T5_active_weapon_P250', 'T5_active_weapon_Tec-9', 'T5_active_weapon_CZ75 Auto', 'T5_active_weapon_Five-SeveN', 'T5_active_weapon_Desert Eagle', 'T5_active_weapon_MAC-10', 'T5_active_weapon_MP9', 'T5_active_weapon_MP7', 'T5_active_weapon_MP5-SD', 'T5_active_weapon_UMP-45', 'T5_active_weapon_PP-Bizon', 'T5_active_weapon_P90', 'T5_active_weapon_Nova', 'T5_active_weapon_XM1014', 'T5_active_weapon_Sawed-Off', 'T5_active_weapon_MAG-7', 'T5_active_weapon_M249', 'T5_active_weapon_Negev', 'T5_active_weapon_FAMAS', 'T5_active_weapon_Galil AR', 'T5_active_weapon_AK-47', 'T5_active_weapon_M4A4', 'T5_active_weapon_M4A1-S', 'T5_active_weapon_SG 553', 'T5_active_weapon_AUG', 'T5_active_weapon_SSG 08', 'T5_active_weapon_AWP', 'T5_active_weapon_G3SG1', 'T5_active_weapon_SCAR-20', 'T5_active_weapon_HE Grenade', 'T5_active_weapon_Flashbang', 'T5_active_weapon_Smoke Grenade', 'T5_active_weapon_Incendiary Grenade', 'T5_active_weapon_Molotov', 'T5_active_weapon_Decoy Grenade',
            'T5_hltv_rating_2.0', 'T5_hltv_DPR', 'T5_hltv_KAST', 'T5_hltv_Impact', 'T5_hltv_ADR', 'T5_hltv_KPR', 'T5_hltv_total_kills', 'T5_hltv_HS%', 'T5_hltv_total_deaths', 'T5_hltv_KD_ratio', 'T5_hltv_dmgPR', 'T5_hltv_grenade_dmgPR', 'T5_hltv_maps_played', 'T5_hltv_saved_by_teammatePR', 'T5_hltv_saved_teammatesPR', 'T5_hltv_opening_kill_rating', 'T5_hltv_team_W%_after_opening', 'T5_hltv_opening_kill_in_W_rounds', 'T5_hltv_rating_1.0_all_Career', 'T5_hltv_clutches_1on1_ratio', 'T5_hltv_clutches_won_1on1', 'T5_hltv_clutches_won_1on2', 'T5_hltv_clutches_won_1on3', 'T5_hltv_clutches_won_1on4', 'T5_hltv_clutches_won_1on5', 

            'T6_name', 'T6_X', 'T6_Y', 'T6_Z', 'T6_pitch', 'T6_yaw', 'T6_velocity_X', 'T6_velocity_Y', 'T6_velocity_Z', 'T6_health', 'T6_armor_value', 'T6_active_weapon_ammo', 'T6_total_ammo_left', 'T6_flash_duration', 'T6_balance', 'T6_current_equip_value', 'T6_round_start_equip_value', 
            'T6_is_alive', 'T6_is_CT', 'T6_is_shooting', 'T6_is_crouching', 'T6_is_ducking', 'T6_is_duck_jumping', 'T6_is_walking', 'T6_is_spotted', 'T6_is_scoped', 'T6_is_defusing', 'T6_is_reloading', 'T6_is_in_bombsite',
            'T6_stat_kills', 'T6_stat_HS_kills', 'T6_stat_opening_kills', 'T6_stat_deaths', 'T6_stat_opening_deaths', 'T6_stat_assists', 'T6_stat_flash_assists', 'T6_stat_damage', 'T6_stat_weapon_damage', 'T6_stat_nade_damage', 'T6_stat_survives', 'T6_stat_KPR', 'T6_stat_ADR', 'T6_stat_DPR', 'T6_stat_HS%', 'T6_stat_SPR', 
            'T6_inventory_C4', 'T6_inventory_Taser', 'T6_inventory_USP-S', 'T6_inventory_P2000', 'T6_inventory_Glock-18', 'T6_inventory_Dual Berettas', 'T6_inventory_P250', 'T6_inventory_Tec-9', 'T6_inventory_CZ75 Auto', 'T6_inventory_Five-SeveN', 'T6_inventory_Desert Eagle', 'T6_inventory_MAC-10', 'T6_inventory_MP9', 'T6_inventory_MP7', 'T6_inventory_MP5-SD', 'T6_inventory_UMP-45', 'T6_inventory_PP-Bizon', 'T6_inventory_P90', 'T6_inventory_Nova', 'T6_inventory_XM1014', 'T6_inventory_Sawed-Off', 'T6_inventory_MAG-7', 'T6_inventory_M249', 'T6_inventory_Negev', 'T6_inventory_FAMAS', 'T6_inventory_Galil AR', 'T6_inventory_AK-47', 'T6_inventory_M4A4', 'T6_inventory_M4A1-S', 'T6_inventory_SG 553', 'T6_inventory_AUG', 'T6_inventory_SSG 08', 'T6_inventory_AWP', 'T6_inventory_G3SG1', 'T6_inventory_SCAR-20', 'T6_inventory_HE Grenade', 'T6_inventory_Flashbang', 'T6_inventory_Smoke Grenade', 'T6_inventory_Incendiary Grenade', 'T6_inventory_Molotov', 'T6_inventory_Decoy Grenade',
            'T6_active_weapon_C4', 'T6_active_weapon_Knife', 'T6_active_weapon_Taser', 'T6_active_weapon_USP-S', 'T6_active_weapon_P2000', 'T6_active_weapon_Glock-18', 'T6_active_weapon_Dual Berettas', 'T6_active_weapon_P250', 'T6_active_weapon_Tec-9', 'T6_active_weapon_CZ75 Auto', 'T6_active_weapon_Five-SeveN', 'T6_active_weapon_Desert Eagle', 'T6_active_weapon_MAC-10', 'T6_active_weapon_MP9', 'T6_active_weapon_MP7', 'T6_active_weapon_MP5-SD', 'T6_active_weapon_UMP-45', 'T6_active_weapon_PP-Bizon', 'T6_active_weapon_P90', 'T6_active_weapon_Nova', 'T6_active_weapon_XM1014', 'T6_active_weapon_Sawed-Off', 'T6_active_weapon_MAG-7', 'T6_active_weapon_M249', 'T6_active_weapon_Negev', 'T6_active_weapon_FAMAS', 'T6_active_weapon_Galil AR', 'T6_active_weapon_AK-47', 'T6_active_weapon_M4A4', 'T6_active_weapon_M4A1-S', 'T6_active_weapon_SG 553', 'T6_active_weapon_AUG', 'T6_active_weapon_SSG 08', 'T6_active_weapon_AWP', 'T6_active_weapon_G3SG1', 'T6_active_weapon_SCAR-20', 'T6_active_weapon_HE Grenade', 'T6_active_weapon_Flashbang', 'T6_active_weapon_Smoke Grenade', 'T6_active_weapon_Incendiary Grenade', 'T6_active_weapon_Molotov', 'T6_active_weapon_Decoy Grenade',
            'T6_hltv_rating_2.0', 'T6_hltv_DPR', 'T6_hltv_KAST', 'T6_hltv_Impact', 'T6_hltv_ADR', 'T6_hltv_KPR', 'T6_hltv_total_kills', 'T6_hltv_HS%', 'T6_hltv_total_deaths', 'T6_hltv_KD_ratio', 'T6_hltv_dmgPR', 'T6_hltv_grenade_dmgPR', 'T6_hltv_maps_played', 'T6_hltv_saved_by_teammatePR', 'T6_hltv_saved_teammatesPR', 'T6_hltv_opening_kill_rating', 'T6_hltv_team_W%_after_opening', 'T6_hltv_opening_kill_in_W_rounds', 'T6_hltv_rating_1.0_all_Career', 'T6_hltv_clutches_1on1_ratio', 'T6_hltv_clutches_won_1on1', 'T6_hltv_clutches_won_1on2', 'T6_hltv_clutches_won_1on3', 'T6_hltv_clutches_won_1on4', 'T6_hltv_clutches_won_1on5', 

            'T7_name', 'T7_X', 'T7_Y', 'T7_Z', 'T7_pitch', 'T7_yaw', 'T7_velocity_X', 'T7_velocity_Y', 'T7_velocity_Z', 'T7_health', 'T7_armor_value', 'T7_active_weapon_ammo', 'T7_total_ammo_left', 'T7_flash_duration', 'T7_balance', 'T7_current_equip_value', 'T7_round_start_equip_value', 
            'T7_is_alive', 'T7_is_CT', 'T7_is_shooting', 'T7_is_crouching', 'T7_is_ducking', 'T7_is_duck_jumping', 'T7_is_walking', 'T7_is_spotted', 'T7_is_scoped', 'T7_is_defusing', 'T7_is_reloading', 'T7_is_in_bombsite',
            'T7_stat_kills', 'T7_stat_HS_kills', 'T7_stat_opening_kills', 'T7_stat_deaths', 'T7_stat_opening_deaths', 'T7_stat_assists', 'T7_stat_flash_assists', 'T7_stat_damage', 'T7_stat_weapon_damage', 'T7_stat_nade_damage', 'T7_stat_survives', 'T7_stat_KPR', 'T7_stat_ADR', 'T7_stat_DPR', 'T7_stat_HS%', 'T7_stat_SPR', 
            'T7_inventory_C4', 'T7_inventory_Taser', 'T7_inventory_USP-S', 'T7_inventory_P2000', 'T7_inventory_Glock-18', 'T7_inventory_Dual Berettas', 'T7_inventory_P250', 'T7_inventory_Tec-9', 'T7_inventory_CZ75 Auto', 'T7_inventory_Five-SeveN', 'T7_inventory_Desert Eagle', 'T7_inventory_MAC-10', 'T7_inventory_MP9', 'T7_inventory_MP7', 'T7_inventory_MP5-SD', 'T7_inventory_UMP-45', 'T7_inventory_PP-Bizon', 'T7_inventory_P90', 'T7_inventory_Nova', 'T7_inventory_XM1014', 'T7_inventory_Sawed-Off', 'T7_inventory_MAG-7', 'T7_inventory_M249', 'T7_inventory_Negev', 'T7_inventory_FAMAS', 'T7_inventory_Galil AR', 'T7_inventory_AK-47', 'T7_inventory_M4A4', 'T7_inventory_M4A1-S', 'T7_inventory_SG 553', 'T7_inventory_AUG', 'T7_inventory_SSG 08', 'T7_inventory_AWP', 'T7_inventory_G3SG1', 'T7_inventory_SCAR-20', 'T7_inventory_HE Grenade', 'T7_inventory_Flashbang', 'T7_inventory_Smoke Grenade', 'T7_inventory_Incendiary Grenade', 'T7_inventory_Molotov', 'T7_inventory_Decoy Grenade',
            'T7_active_weapon_C4', 'T7_active_weapon_Knife', 'T7_active_weapon_Taser', 'T7_active_weapon_USP-S', 'T7_active_weapon_P2000', 'T7_active_weapon_Glock-18', 'T7_active_weapon_Dual Berettas', 'T7_active_weapon_P250', 'T7_active_weapon_Tec-9', 'T7_active_weapon_CZ75 Auto', 'T7_active_weapon_Five-SeveN', 'T7_active_weapon_Desert Eagle', 'T7_active_weapon_MAC-10', 'T7_active_weapon_MP9', 'T7_active_weapon_MP7', 'T7_active_weapon_MP5-SD', 'T7_active_weapon_UMP-45', 'T7_active_weapon_PP-Bizon', 'T7_active_weapon_P90', 'T7_active_weapon_Nova', 'T7_active_weapon_XM1014', 'T7_active_weapon_Sawed-Off', 'T7_active_weapon_MAG-7', 'T7_active_weapon_M249', 'T7_active_weapon_Negev', 'T7_active_weapon_FAMAS', 'T7_active_weapon_Galil AR', 'T7_active_weapon_AK-47', 'T7_active_weapon_M4A4', 'T7_active_weapon_M4A1-S', 'T7_active_weapon_SG 553', 'T7_active_weapon_AUG', 'T7_active_weapon_SSG 08', 'T7_active_weapon_AWP', 'T7_active_weapon_G3SG1', 'T7_active_weapon_SCAR-20', 'T7_active_weapon_HE Grenade', 'T7_active_weapon_Flashbang', 'T7_active_weapon_Smoke Grenade', 'T7_active_weapon_Incendiary Grenade', 'T7_active_weapon_Molotov', 'T7_active_weapon_Decoy Grenade',
            'T7_hltv_rating_2.0', 'T7_hltv_DPR', 'T7_hltv_KAST', 'T7_hltv_Impact', 'T7_hltv_ADR', 'T7_hltv_KPR', 'T7_hltv_total_kills', 'T7_hltv_HS%', 'T7_hltv_total_deaths', 'T7_hltv_KD_ratio', 'T7_hltv_dmgPR', 'T7_hltv_grenade_dmgPR', 'T7_hltv_maps_played', 'T7_hltv_saved_by_teammatePR', 'T7_hltv_saved_teammatesPR', 'T7_hltv_opening_kill_rating', 'T7_hltv_team_W%_after_opening', 'T7_hltv_opening_kill_in_W_rounds', 'T7_hltv_rating_1.0_all_Career', 'T7_hltv_clutches_1on1_ratio', 'T7_hltv_clutches_won_1on1', 'T7_hltv_clutches_won_1on2', 'T7_hltv_clutches_won_1on3', 'T7_hltv_clutches_won_1on4', 'T7_hltv_clutches_won_1on5', 

            'T8_name', 'T8_X', 'T8_Y', 'T8_Z', 'T8_pitch', 'T8_yaw', 'T8_velocity_X', 'T8_velocity_Y', 'T8_velocity_Z', 'T8_health', 'T8_armor_value', 'T8_active_weapon_ammo', 'T8_total_ammo_left', 'T8_flash_duration', 'T8_balance', 'T8_current_equip_value', 'T8_round_start_equip_value', 
            'T8_is_alive', 'T8_is_CT', 'T8_is_shooting', 'T8_is_crouching', 'T8_is_ducking', 'T8_is_duck_jumping', 'T8_is_walking', 'T8_is_spotted', 'T8_is_scoped', 'T8_is_defusing', 'T8_is_reloading', 'T8_is_in_bombsite',
            'T8_stat_kills', 'T8_stat_HS_kills', 'T8_stat_opening_kills', 'T8_stat_deaths', 'T8_stat_opening_deaths', 'T8_stat_assists', 'T8_stat_flash_assists', 'T8_stat_damage', 'T8_stat_weapon_damage', 'T8_stat_nade_damage', 'T8_stat_survives', 'T8_stat_KPR', 'T8_stat_ADR', 'T8_stat_DPR', 'T8_stat_HS%', 'T8_stat_SPR', 
            'T8_inventory_C4', 'T8_inventory_Taser', 'T8_inventory_USP-S', 'T8_inventory_P2000', 'T8_inventory_Glock-18', 'T8_inventory_Dual Berettas', 'T8_inventory_P250', 'T8_inventory_Tec-9', 'T8_inventory_CZ75 Auto', 'T8_inventory_Five-SeveN', 'T8_inventory_Desert Eagle', 'T8_inventory_MAC-10', 'T8_inventory_MP9', 'T8_inventory_MP7', 'T8_inventory_MP5-SD', 'T8_inventory_UMP-45', 'T8_inventory_PP-Bizon', 'T8_inventory_P90', 'T8_inventory_Nova', 'T8_inventory_XM1014', 'T8_inventory_Sawed-Off', 'T8_inventory_MAG-7', 'T8_inventory_M249', 'T8_inventory_Negev', 'T8_inventory_FAMAS', 'T8_inventory_Galil AR', 'T8_inventory_AK-47', 'T8_inventory_M4A4', 'T8_inventory_M4A1-S', 'T8_inventory_SG 553', 'T8_inventory_AUG', 'T8_inventory_SSG 08', 'T8_inventory_AWP', 'T8_inventory_G3SG1', 'T8_inventory_SCAR-20', 'T8_inventory_HE Grenade', 'T8_inventory_Flashbang', 'T8_inventory_Smoke Grenade', 'T8_inventory_Incendiary Grenade', 'T8_inventory_Molotov', 'T8_inventory_Decoy Grenade',
            'T8_active_weapon_C4', 'T8_active_weapon_Knife', 'T8_active_weapon_Taser', 'T8_active_weapon_USP-S', 'T8_active_weapon_P2000', 'T8_active_weapon_Glock-18', 'T8_active_weapon_Dual Berettas', 'T8_active_weapon_P250', 'T8_active_weapon_Tec-9', 'T8_active_weapon_CZ75 Auto', 'T8_active_weapon_Five-SeveN', 'T8_active_weapon_Desert Eagle', 'T8_active_weapon_MAC-10', 'T8_active_weapon_MP9', 'T8_active_weapon_MP7', 'T8_active_weapon_MP5-SD', 'T8_active_weapon_UMP-45', 'T8_active_weapon_PP-Bizon', 'T8_active_weapon_P90', 'T8_active_weapon_Nova', 'T8_active_weapon_XM1014', 'T8_active_weapon_Sawed-Off', 'T8_active_weapon_MAG-7', 'T8_active_weapon_M249', 'T8_active_weapon_Negev', 'T8_active_weapon_FAMAS', 'T8_active_weapon_Galil AR', 'T8_active_weapon_AK-47', 'T8_active_weapon_M4A4', 'T8_active_weapon_M4A1-S', 'T8_active_weapon_SG 553', 'T8_active_weapon_AUG', 'T8_active_weapon_SSG 08', 'T8_active_weapon_AWP', 'T8_active_weapon_G3SG1', 'T8_active_weapon_SCAR-20', 'T8_active_weapon_HE Grenade', 'T8_active_weapon_Flashbang', 'T8_active_weapon_Smoke Grenade', 'T8_active_weapon_Incendiary Grenade', 'T8_active_weapon_Molotov', 'T8_active_weapon_Decoy Grenade',
            'T8_hltv_rating_2.0', 'T8_hltv_DPR', 'T8_hltv_KAST', 'T8_hltv_Impact', 'T8_hltv_ADR', 'T8_hltv_KPR', 'T8_hltv_total_kills', 'T8_hltv_HS%', 'T8_hltv_total_deaths', 'T8_hltv_KD_ratio', 'T8_hltv_dmgPR', 'T8_hltv_grenade_dmgPR', 'T8_hltv_maps_played', 'T8_hltv_saved_by_teammatePR', 'T8_hltv_saved_teammatesPR', 'T8_hltv_opening_kill_rating', 'T8_hltv_team_W%_after_opening', 'T8_hltv_opening_kill_in_W_rounds', 'T8_hltv_rating_1.0_all_Career', 'T8_hltv_clutches_1on1_ratio', 'T8_hltv_clutches_won_1on1', 'T8_hltv_clutches_won_1on2', 'T8_hltv_clutches_won_1on3', 'T8_hltv_clutches_won_1on4', 'T8_hltv_clutches_won_1on5', 

            'T9_name', 'T9_X', 'T9_Y', 'T9_Z', 'T9_pitch', 'T9_yaw', 'T9_velocity_X', 'T9_velocity_Y', 'T9_velocity_Z', 'T9_health', 'T9_armor_value', 'T9_active_weapon_ammo', 'T9_total_ammo_left', 'T9_flash_duration', 'T9_balance', 'T9_current_equip_value', 'T9_round_start_equip_value', 
            'T9_is_alive', 'T9_is_CT', 'T9_is_shooting', 'T9_is_crouching', 'T9_is_ducking', 'T9_is_duck_jumping', 'T9_is_walking', 'T9_is_spotted', 'T9_is_scoped', 'T9_is_defusing', 'T9_is_reloading', 'T9_is_in_bombsite',
            'T9_stat_kills', 'T9_stat_HS_kills', 'T9_stat_opening_kills', 'T9_stat_deaths', 'T9_stat_opening_deaths', 'T9_stat_assists', 'T9_stat_flash_assists', 'T9_stat_damage', 'T9_stat_weapon_damage', 'T9_stat_nade_damage', 'T9_stat_survives', 'T9_stat_KPR', 'T9_stat_ADR', 'T9_stat_DPR', 'T9_stat_HS%', 'T9_stat_SPR', 
            'T9_inventory_C4', 'T9_inventory_Taser', 'T9_inventory_USP-S', 'T9_inventory_P2000', 'T9_inventory_Glock-18', 'T9_inventory_Dual Berettas', 'T9_inventory_P250', 'T9_inventory_Tec-9', 'T9_inventory_CZ75 Auto', 'T9_inventory_Five-SeveN', 'T9_inventory_Desert Eagle', 'T9_inventory_MAC-10', 'T9_inventory_MP9', 'T9_inventory_MP7', 'T9_inventory_MP5-SD', 'T9_inventory_UMP-45', 'T9_inventory_PP-Bizon', 'T9_inventory_P90', 'T9_inventory_Nova', 'T9_inventory_XM1014', 'T9_inventory_Sawed-Off', 'T9_inventory_MAG-7', 'T9_inventory_M249', 'T9_inventory_Negev', 'T9_inventory_FAMAS', 'T9_inventory_Galil AR', 'T9_inventory_AK-47', 'T9_inventory_M4A4', 'T9_inventory_M4A1-S', 'T9_inventory_SG 553', 'T9_inventory_AUG', 'T9_inventory_SSG 08', 'T9_inventory_AWP', 'T9_inventory_G3SG1', 'T9_inventory_SCAR-20', 'T9_inventory_HE Grenade', 'T9_inventory_Flashbang', 'T9_inventory_Smoke Grenade', 'T9_inventory_Incendiary Grenade', 'T9_inventory_Molotov', 'T9_inventory_Decoy Grenade',
            'T9_active_weapon_C4', 'T9_active_weapon_Knife', 'T9_active_weapon_Taser', 'T9_active_weapon_USP-S', 'T9_active_weapon_P2000', 'T9_active_weapon_Glock-18', 'T9_active_weapon_Dual Berettas', 'T9_active_weapon_P250', 'T9_active_weapon_Tec-9', 'T9_active_weapon_CZ75 Auto', 'T9_active_weapon_Five-SeveN', 'T9_active_weapon_Desert Eagle', 'T9_active_weapon_MAC-10', 'T9_active_weapon_MP9', 'T9_active_weapon_MP7', 'T9_active_weapon_MP5-SD', 'T9_active_weapon_UMP-45', 'T9_active_weapon_PP-Bizon', 'T9_active_weapon_P90', 'T9_active_weapon_Nova', 'T9_active_weapon_XM1014', 'T9_active_weapon_Sawed-Off', 'T9_active_weapon_MAG-7', 'T9_active_weapon_M249', 'T9_active_weapon_Negev', 'T9_active_weapon_FAMAS', 'T9_active_weapon_Galil AR', 'T9_active_weapon_AK-47', 'T9_active_weapon_M4A4', 'T9_active_weapon_M4A1-S', 'T9_active_weapon_SG 553', 'T9_active_weapon_AUG', 'T9_active_weapon_SSG 08', 'T9_active_weapon_AWP', 'T9_active_weapon_G3SG1', 'T9_active_weapon_SCAR-20', 'T9_active_weapon_HE Grenade', 'T9_active_weapon_Flashbang', 'T9_active_weapon_Smoke Grenade', 'T9_active_weapon_Incendiary Grenade', 'T9_active_weapon_Molotov', 'T9_active_weapon_Decoy Grenade',
            'T9_hltv_rating_2.0', 'T9_hltv_DPR', 'T9_hltv_KAST', 'T9_hltv_Impact', 'T9_hltv_ADR', 'T9_hltv_KPR', 'T9_hltv_total_kills', 'T9_hltv_HS%', 'T9_hltv_total_deaths', 'T9_hltv_KD_ratio', 'T9_hltv_dmgPR', 'T9_hltv_grenade_dmgPR', 'T9_hltv_maps_played', 'T9_hltv_saved_by_teammatePR', 'T9_hltv_saved_teammatesPR', 'T9_hltv_opening_kill_rating', 'T9_hltv_team_W%_after_opening', 'T9_hltv_opening_kill_in_W_rounds', 'T9_hltv_rating_1.0_all_Career', 'T9_hltv_clutches_1on1_ratio', 'T9_hltv_clutches_won_1on1', 'T9_hltv_clutches_won_1on2', 'T9_hltv_clutches_won_1on3', 'T9_hltv_clutches_won_1on4', 'T9_hltv_clutches_won_1on5', 

            
            'numerical_match_id', 'match_id', 'tick', 'round', 'time', 'remaining_time', 'freeze_end', 'end', 'CT_wins', 
            'CT_alive_num', 'T_alive_num', 'CT_total_hp', 'T_total_hp', 'CT_equipment_value', 'T_equipment_value',  'CT_losing_streak', 'T_losing_streak',
            'is_bomb_dropped', 'is_bomb_being_planted', 'is_bomb_being_defused', 'is_bomb_defused', 'is_bomb_planted_at_A_site', 'is_bomb_planted_at_B_site',
            'bomb_X', 'bomb_Y', 'bomb_Z', 'bomb_mx_pos1', 'bomb_mx_pos2', 'bomb_mx_pos3', 'bomb_mx_pos4', 'bomb_mx_pos5', 'bomb_mx_pos6', 'bomb_mx_pos7', 'bomb_mx_pos8', 'bomb_mx_pos9', 
            'smokes_active', 'infernos_active'
        ]

        # Rearrange the column order
        team_1_ct = team_1_ct[col_order]
        team_2_ct = team_2_ct[col_order]

        # Concatenate the two dataframes
        renamed_df = pd.concat([team_1_ct, team_2_ct])

        return renamed_df



    # 14. Build column dictionary
    def _FINAL_build_dictionary(self, df):

        # Get the numerical columns
        numeric_cols = [col for col in df.columns if '_name' not in col and col not in ['match_id', 'smokes_active', 'infernos_active']]

        # Create dictionary dataset
        df_dict = pd.DataFrame(data={
            'column': numeric_cols, 
            'min': df[numeric_cols].min().values, 
            'max': df[numeric_cols].max().values
        })

        return df_dict