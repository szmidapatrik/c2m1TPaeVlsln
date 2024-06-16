import pandas as pd
import numpy as np
import random


class TabularGraphDataCreator:



    # INPUT
    # Folder path constants
    MATCH_FILE_ID = None
    TABULAR_DATA_FOLDER_PATH = None
    PLAYER_STATS_DATA_PATH = None
    MISSING_PLAYER_STATS_DATA_PATH = None

    
    # Optional variables
    tick_number = 1
    add_numerical_match_id = False
    numerical_match_id = None
    group_players_by_side = True
    vary_player_permutation = False
    num_permutations_per_round = 1



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    def format_match_data(
        self,
        match_file_name: str,
        tabular_data_folder_path: str, 
        player_stats_data_path: str, 
        missing_player_stats_data_path: str,
        tick_number: int = 1,
        add_numerical_match_id: bool = False,
        numerical_match_id: int = None,
        vary_player_permutation: bool = False,
        num_permutations_per_round: int = 1,
        group_players_by_side: bool = True,
    ):
        """
        Formats the match data and creates the tabular game-snapshot dataset. Parameters:
            - match_file_name: name of the match file,
            - tabular_data_folder_path: folder path of the parsed data,
            - player_stats_data_path: path of the player statistics data,
            - output_folder_path: folder path of the output,
            - missing_player_stats_data_path (optional): path of the missing player statistics data,
            - tick_number (optional): parse tick rate.
            - vary_player_permutation (optional): increases dataset size by creating copies of the rounds with varied player permutation. Default is False.
            - group_players_by_side (optional): group players by side. Default is True.
        """

        # INPUT
        self.MATCH_FILE_ID = match_file_name
        self.TABULAR_DATA_FOLDER_PATH = tabular_data_folder_path
        self.PLAYER_STATS_DATA_PATH = player_stats_data_path
        self.MISSING_PLAYER_STATS_DATA_PATH = missing_player_stats_data_path

        # Other variables
        self.tick_number = tick_number
        self.add_numerical_match_id = add_numerical_match_id
        self.numerical_match_id = numerical_match_id
        self.vary_player_permutation = vary_player_permutation
        self.num_permutations_per_round = num_permutations_per_round
        self.group_players_by_side = group_players_by_side


        # 1.
        pf, kills, rounds, bombEvents, damages = self.__INIT_get_needed_dataframes__()
        # 2.
        pf, kills, rounds = self.__PLAYER_calculate_ingame_features_from_needed_dataframes__(pf, kills, rounds, damages)
        # 3.
        pf = self.__PLAYER_get_activeWeapon_dummies__(pf)
        # 4.
        players = self.__PLAYER_player_dataset_create__(pf, self.tick_number)
        # 5.
        players = self.__PLAYER_get_player_overall_statistics_without_inferno__(players)
        # 6.
        tabular_df = self.__TABULAR_create_overall_and_player_tabular_dataset__(players, rounds, self.MATCH_FILE_ID)
        # 7.
        tabular_df = self.__TABULAR_add_bomb_info_to_dataset__(tabular_df, bombEvents)
        # 8.
        tabular_df = self.__TABULAR_calculate_time_from_tick__(tabular_df)
        # 9.
        tabular_df = self.__TABULAR_add_numerical_match_id__(tabular_df)
        # 10.
        tabular_df = self.__TABULAR_bombsite_3x3_matrix_split_for_bomb_pos_feature__(tabular_df)
        # 11.
        if vary_player_permutation:
            tabular_df = self.__TABULAR_vary_player_permutation__(tabular_df, self.num_permutations_per_round)
        # 12.
        if group_players_by_side:
            tabular_df = self.__TABULAR_refactor_player_columns_to_CT_T__(tabular_df)

        return tabular_df





    def create_missing_player_stats_data(self, player_stats_data_path, length=1000):
        """
        Creates a dataset filled with fictive players with average statistics. Useful for missing data imputation.
        """

        self.PLAYER_STATS_DATA_PATH = player_stats_data_path

        mpdf = pd.read_csv(self.PLAYER_STATS_DATA_PATH)
        mpdf = mpdf.drop_duplicates()

        # Store the numerical columns in an array
        numerical_cols = mpdf.select_dtypes(include=[np.number]).columns.tolist()

        # Create a dictionary to store the min and max values of the numerical columns
        dist_values = {}
        for col in numerical_cols:
            dist_values[col] = [mpdf[col].mode().min(), (mpdf[col].max() - mpdf[col].min()) / 150]

        # Create a fictive player with average statistics
        fictive_player = {}
        for col in numerical_cols:
            fictive_player[col] = mpdf[col].mode()

        # Create a DataFrame with the fictive player
        fictive_player_df = pd.DataFrame(fictive_player, index=[0])
        fictive_player_df['player_name'] = 'anonim_pro'
        fictive_player_df = pd.concat([fictive_player_df]*length, ignore_index=True)

        # Create a DataFrame with the fictive player repeated *length* times and with random values
        for col in numerical_cols:
            fictive_player_df[col] = np.random.normal(dist_values[col][0], dist_values[col][1], size=length)
            if col not in ['KD_ratio', 'KD_diff']:
                fictive_player_df[col] = fictive_player_df[col].abs()
            if col in ['total_deaths', 'maps_played', 'rounds_played', 'rounds_with_kils', 'KD_diff', 'total_opening_kills', 'total_opening_deaths', 
                       '0_kill_rounds', '1_kill_rounds', '2_kill_rounds', '3_kill_rounds', '4_kill_rounds', '5_kill_rounds',
                       'rifle_kills', 'sniper_kills', 'smg_kills', 'pistol_kills', 'grenade_kills', 'other_kills', 'rating_2.0_1+', 'rating_2.0_1+_streak', 
                       'clutches_won_1on1', 'clutches_lost_1on1', 'clutches_won_1on2', 'clutches_won_1on3', 'clutches_won_1on4', 'clutches_won_1on5']:
                fictive_player_df[col] = fictive_player_df[col].apply(lambda x: int(x))

        fictive_player_df['clutches_1on1_ratio'] = fictive_player_df['clutches_won_1on1'] / fictive_player_df['clutches_lost_1on1']

        return fictive_player_df




    # --------------------------------------------------------------------------------------------

    # 1. Get needed dataframes
    def __INIT_get_needed_dataframes__(self):

        # Read dataframes
        playerFrames = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/playerFrames/' + self.MATCH_FILE_ID)
        kills = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH +'/kills/' + self.MATCH_FILE_ID)
        rounds = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH +'/rounds/' + self.MATCH_FILE_ID)
        bombEvents = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/bombEvents/' + self.MATCH_FILE_ID)
        damages = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/damages/' + self.MATCH_FILE_ID)

        # Filter columns
        rounds = rounds[['roundNum', 'tScore', "ctScore" ,'endTScore', 'endCTScore']]
        pf = playerFrames[['tick', 'roundNum', 'seconds', 'side', 'name', 'x', 'y', 'z','eyeX', 'eyeY', 'eyeZ', 'velocityX', 'velocityY', 'velocityZ',
            'hp', 'armor', 'activeWeapon','flashGrenades', 'smokeGrenades', 'heGrenades', 'totalUtility', 'isAlive', 'isReloading', 'isBlinded', 'isDucking',
            'isDefusing', 'isPlanting', 'isUnknown', 'isScoped', 'equipmentValue', 'equipmentValueRoundStart', 'hasHelmet','hasDefuse', 'hasBomb']]
        
        return pf, kills, rounds, bombEvents, damages


    # 2. Calculate ingame player statistics
    def __PLAYER_calculate_ingame_features_from_needed_dataframes__(self, pf, kills, rounds, damages):
    
        # Merge playerFrames with rounds
        pf = pf.merge(rounds, on='roundNum')

        # Format CT information
        pf['isCT'] = pf.apply(lambda x: 1 if x['side'] == 'CT' else 0, axis=1)

        # Kill stats
        pf['stat_kills'] = 0
        pf['stat_HSK'] = 0
        pf['stat_openKills'] = 0
        pf['stat_tradeKills'] = 0
        # Death stats
        pf['stat_deaths'] = 0
        pf['stat_openDeaths'] = 0
        # Assist stats
        pf['stat_assists'] = 0
        pf['stat_flashAssists'] = 0
        # Damage stats
        pf['stat_damage'] = 0
        pf['stat_weaponDamage'] = 0
        pf['stat_nadeDamage'] = 0

        # Setting kill-stats
        for _, row in kills.iterrows():

            # Kills
            pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_kills'] += 1
            # HS-kills
            if row['isHeadshot']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_HSK'] += 1
            # Opening-kills
            if row['isFirstKill']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_openKills'] += 1
            # Trading-kills
            if row['isTrade']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_tradeKills'] += 1
            # Deaths
            pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['victimName']), 'stat_deaths'] += 1
            # Opening deaths
            if row['isFirstKill']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['victimName']), 'stat_openDeaths'] += 1
            # Assists
            if pd.notna(row['assisterName']):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['assisterName']), 'stat_assists'] += 1
            # Flash assists
            if row['victimBlinded'] and row['flashThrowerTeam'] != row['victimTeam']:
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['flashThrowerTeam']), 'stat_flashAssists'] += 1

        # Setting damage-stats
        for _, row in damages.iterrows():

            # All Damage
            if (row['isFriendlyFire'] == False):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_damage'] += row['hpDamageTaken']
            # Weapon Damage
            if (row['isFriendlyFire'] == False) and (row['weaponClass'] != "Grenade" and row['weaponClass'] != "Equipment"):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_weaponDamage'] += row['hpDamageTaken']
            # Nade Damage
            if (row['isFriendlyFire'] == False) and (row['weaponClass'] == "Grenade"):
                pf.loc[(pf['tick'] >= row['tick']) & (pf['name'] == row['attackerName']), 'stat_nadeDamage'] += row['hpDamageTaken']
            
        return pf, kills, rounds
    

    # 3. Handle active weapon column
    def __PLAYER_get_activeWeapon_dummies__(self, pf):
    
        # Active weapons
        active_weapons = [
            # Other
            'activeWeapon_C4', 'activeWeapon_Knife', 'activeWeapon_Taser',
            # Pistols
            'activeWeapon_USP-S', 'activeWeapon_P2000', 'activeWeapon_Glock-18', 'activeWeapon_Dual Berettas', 'activeWeapon_P250', 'activeWeapon_Tec-9', 'activeWeapon_CZ75 Auto', 'activeWeapon_Five-SeveN', 'activeWeapon_Desert Eagle',
            # SMGs
            'activeWeapon_MAC-10', 'activeWeapon_MP9', 'activeWeapon_MP7', 'activeWeapon_MP5-SD', 'activeWeapon_UMP-45', 'activeWeapon_PP-Bizon', 'activeWeapon_P90',
            # Heavy
            'activeWeapon_Nova', 'activeWeapon_XM1014', 'activeWeapon_Sawed-Off', 'activeWeapon_MAG-7', 'activeWeapon_M249', 'activeWeapon_Negev',
            # Rifles
            'activeWeapon_FAMAS', 'activeWeapon_Galil AR', 'activeWeapon_AK-47', 'activeWeapon_M4A4', 'activeWeapon_M4A1', 'activeWeapon_SG 553', 'activeWeapon_AUG', 'activeWeapon_SSG 08', 'activeWeapon_AWP', 'activeWeapon_G3SG1', 'activeWeapon_SCAR-20',
            # Grenades
            'activeWeapon_HE Grenade', 'activeWeapon_Flashbang', 'activeWeapon_Smoke Grenade', 'activeWeapon_Incendiary Grenade', 'activeWeapon_Molotov', 'activeWeapon_Decoy Grenade'
        ]

        # Create dummie cols
        df_dummies = pd.get_dummies(pf['activeWeapon'], prefix="activeWeapon",drop_first=False)
        dummies = pd.DataFrame()
        for col in active_weapons:
            if col not in df_dummies.columns:
                dummies[col] = np.zeros(len(df_dummies))
            else:
                dummies[col] = df_dummies[col]
        
        dummies = dummies*1
        pf = pf.merge(dummies, left_index = True, right_index = True, how = 'left')
        
        return pf
    

    # 4. Create player dataset
    def __PLAYER_player_dataset_create__(self, pf, tick_number = 1):
    
        startAsCTPlayerNames = pf[(pf['side'] == 'CT') & (pf['roundNum'] == 1)]['name'].unique()
        startAsTPlayerNames = pf[(pf['side'] == 'T') & (pf['roundNum'] == 1)]['name'].unique()
        players = {}

        # Team 1: start on CT side
        players[0] = pf[pf['name'] == startAsCTPlayerNames[0]].iloc[::tick_number].copy()
        players[1] = pf[pf['name'] == startAsCTPlayerNames[1]].iloc[::tick_number].copy()
        players[2] = pf[pf['name'] == startAsCTPlayerNames[2]].iloc[::tick_number].copy()
        players[3] = pf[pf['name'] == startAsCTPlayerNames[3]].iloc[::tick_number].copy()
        players[4] = pf[pf['name'] == startAsCTPlayerNames[4]].iloc[::tick_number].copy()

        # Team 2: start on T side
        players[5] = pf[pf['name'] == startAsTPlayerNames[0]].iloc[::tick_number].copy()
        players[6] = pf[pf['name'] == startAsTPlayerNames[1]].iloc[::tick_number].copy()
        players[7] = pf[pf['name'] == startAsTPlayerNames[2]].iloc[::tick_number].copy()
        players[8] = pf[pf['name'] == startAsTPlayerNames[3]].iloc[::tick_number].copy()
        players[9] = pf[pf['name'] == startAsTPlayerNames[4]].iloc[::tick_number].copy()
        
        return players
    

    # 5. Insert universal player statistics into player dataset
    def __EXT_insert_columns_into_player_dataframes__(self, stat_df, players_df):
        for col in stat_df.columns:
            if col != 'player_name':
                players_df[col] = stat_df.loc[stat_df['player_name'] == players_df['name'].iloc[0]][col].iloc[0]
        return players_df

    def __PLAYER_get_player_overall_statistics_without_inferno__(self, players):
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
                stats.rename(columns={col: "overall_" + col}, inplace=True)
        
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
                        mpdf.rename(columns={col: "overall_" + col}, inplace=True)
                        
                # If the missing_players_df contains the player related informations, do the merge
                if len(mpdf.loc[mpdf['player_name'] == players[idx]['name'].iloc[0]]) == 1:
                    players[idx] = self.__EXT_insert_columns_into_player_dataframes__(mpdf, players[idx])

                # Else get imputed values for the player from missing_players_df and do the merge
                else:
                    first_anonim_pro_index = mpdf.index[mpdf['player_name'] == 'anonim_pro'].min()
                    mpdf.at[first_anonim_pro_index, 'player_name'] = players[idx]['name'].iloc[0]
                    players[idx] = self.__EXT_insert_columns_into_player_dataframes__(mpdf, players[idx])
                    
                    # Reverse the column renaming - remove the 'overall_' prefix
                    for col in mpdf.columns:
                        if col.startswith('overall_'):
                            new_col = col[len('overall_'):]
                            mpdf.rename(columns={col: new_col}, inplace=True)

                    mpdf.to_csv(self.MISSING_PLAYER_STATS_DATA_PATH, index=False)
            
        return players
    

    # 6. Create tabular dataset - first version (1 row - 1 graph)
    def __EXT_calculate_ct_equipment_value__(self, row):
        if row['player0_isCT']:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __EXT_calculate_t_equipment_value__(self, row):
        if row['player0_isCT'] == False:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __EXT_calculate_ct_total_hp__(self, row):
        if row['player0_isCT']:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __EXT_calculate_t_total_hp__(self, row):
        if row['player0_isCT'] == False:
            return row[['player0_hp', 'player1_hp', 'player2_hp', 'player3_hp', 'player4_hp']].sum()
        else:
            return row[['player5_hp', 'player6_hp', 'player7_hp', 'player8_hp', 'player9_hp']].sum()

    def __TABULAR_create_overall_and_player_tabular_dataset__(self, players, rounds, match_id):
        """
        Creates the first version of the dataset for the graph model.
        """

        # Copy players object
        graph_players = {}
        for idx in range(0,len(players)):
            graph_players[idx] = players[idx].copy()

        colsNotToRename = ['tick', 'roundNum', 'seconds']

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

        graph_data = graph_data.merge(rounds, on=['roundNum'])
        graph_data['CT_winsRound'] = graph_data.apply(lambda x: 1 if (x['endCTScore'] > x['ctScore']) else 0, axis=1)

        graph_data['player0_equi_val_alive'] = graph_data['player0_equipmentValue'] * graph_data['player0_isAlive']
        graph_data['player1_equi_val_alive'] = graph_data['player1_equipmentValue'] * graph_data['player1_isAlive']
        graph_data['player2_equi_val_alive'] = graph_data['player2_equipmentValue'] * graph_data['player2_isAlive']
        graph_data['player3_equi_val_alive'] = graph_data['player3_equipmentValue'] * graph_data['player3_isAlive']
        graph_data['player4_equi_val_alive'] = graph_data['player4_equipmentValue'] * graph_data['player4_isAlive']
        graph_data['player5_equi_val_alive'] = graph_data['player5_equipmentValue'] * graph_data['player5_isAlive']
        graph_data['player6_equi_val_alive'] = graph_data['player6_equipmentValue'] * graph_data['player6_isAlive']
        graph_data['player7_equi_val_alive'] = graph_data['player7_equipmentValue'] * graph_data['player7_isAlive']
        graph_data['player8_equi_val_alive'] = graph_data['player8_equipmentValue'] * graph_data['player8_isAlive']
        graph_data['player9_equi_val_alive'] = graph_data['player9_equipmentValue'] * graph_data['player9_isAlive']

        graph_data['CT_aliveNum'] = graph_data[['player0_isAlive','player1_isAlive','player2_isAlive','player3_isAlive','player4_isAlive']].sum(axis=1)
        graph_data['T_aliveNum'] = graph_data[['player5_isAlive','player6_isAlive','player7_isAlive','player8_isAlive','player9_isAlive']].sum(axis=1)

        graph_data['CT_equipmentValue'] = graph_data.apply(self.__EXT_calculate_ct_equipment_value__, axis=1)
        graph_data['T_equipmentValue'] = graph_data.apply(self.__EXT_calculate_t_equipment_value__, axis=1)
        
        graph_data['CT_totalHP'] = graph_data.apply(self.__EXT_calculate_ct_equipment_value__, axis=1)
        graph_data['T_totalHP'] = graph_data.apply(self.__EXT_calculate_t_equipment_value__, axis=1)

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
        
        del graph_data['player0_activeWeapon']
        del graph_data['player1_activeWeapon']
        del graph_data['player2_activeWeapon']
        del graph_data['player3_activeWeapon']
        del graph_data['player4_activeWeapon']
        del graph_data['player5_activeWeapon']
        del graph_data['player6_activeWeapon']
        del graph_data['player7_activeWeapon']
        del graph_data['player8_activeWeapon']
        del graph_data['player9_activeWeapon']

        del graph_data['player0_side']
        del graph_data['player1_side']
        del graph_data['player2_side']
        del graph_data['player3_side']
        del graph_data['player4_side']
        del graph_data['player5_side']
        del graph_data['player6_side']
        del graph_data['player7_side']
        del graph_data['player8_side']
        del graph_data['player9_side']

        # Create a DataFrame with a single column for match_id
        match_id_df = pd.DataFrame({'match_id': str(match_id)}, index=graph_data.index)
        graph_data_concatenated = pd.concat([graph_data, match_id_df], axis=1)

        return graph_data_concatenated


    # 7. Add bomb information to the dataset
    def __TABULAR_add_bomb_info_to_dataset__(self, tabular_df, bombdf):
        
        # Poor performance
        # tabular_df['is_bomb_being_planted'] = 0
        # tabular_df['is_bomb_being_defused'] = 0
        # tabular_df['is_bomb_defused'] = 0
        # tabular_df['is_bomb_planted_at_A_site'] = 0
        # tabular_df['is_bomb_planted_at_B_site'] = 0
        # tabular_df['bomb_X'] = 0.0
        # tabular_df['bomb_Y'] = 0.0
        # tabular_df['bomb_Z'] = 0.0

        new_columns = pd.DataFrame({
            'is_bomb_being_planted': 0,
            'is_bomb_being_defused': 0,
            'is_bomb_defused': 0,
            'is_bomb_planted_at_A_site': 0,
            'is_bomb_planted_at_B_site': 0,
            'bomb_X': 0.0,
            'bomb_Y': 0.0,
            'bomb_Z': 0.0
        }, index=tabular_df.index)

        tabular_df = pd.concat([tabular_df, new_columns], axis=1)

        for _, row in bombdf.iterrows():
            if (row['bombAction'] == 'plant_begin'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_planted'] = 1

            if (row['bombAction'] == 'plant_abort'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_planted'] = 0

            if (row['bombAction'] == 'plant'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_planted'] = 0
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_planted_at_A_site'] = 1 if row['bombSite'] == 'A' else 0
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_planted_at_B_site'] = 1 if row['bombSite'] == 'B' else 0
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'bomb_X'] = row['playerX']
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'bomb_Y'] = row['playerY']
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'bomb_Z'] = row['playerZ']

            if (row['bombAction'] == 'defuse_start'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_defused'] = 1

            if (row['bombAction'] == 'defuse_aborted'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_defused'] = 0

            if (row['bombAction'] == 'defuse'):
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_being_defused'] = 0
                tabular_df.loc[(tabular_df['roundNum'] == row['roundNum']) & (tabular_df['tick'] >= row['tick']), 'is_bomb_defused'] = 1

        return tabular_df
    

    # 8. Calculate accurate time feature
    def __TABULAR_calculate_time_from_tick__(self, tabular_df):

        # Get round start tick and use it to calculate the time remaining in the round
        roundStartTick = tabular_df[['match_id', 'roundNum', 'tick']].drop_duplicates(subset=['match_id', 'roundNum']).rename(columns={"tick": "roundStartTick"}).copy()
        tabular_df = tabular_df.merge(roundStartTick, on=['match_id', 'roundNum'])
        # Poor performance
        # tabular_df['sec'] = (tabular_df['tick'] - tabular_df['roundStartTick']) / 128
        # tabular_df['time_remaining'] = 115 - tabular_df['sec']

        new_columns = pd.DataFrame({
            'sec': (tabular_df['tick'] - tabular_df['roundStartTick']) / 128,
            'time_remaining': 115 - (tabular_df['tick'] - tabular_df['roundStartTick']) / 128,
        }, index=tabular_df.index)

        tabular_df = pd.concat([tabular_df, new_columns], axis=1)

        # Drop unnecessary columns
        del tabular_df['roundStartTick']
        del tabular_df['sec']           # Stored in remaining time feature
        del tabular_df['seconds']       # Stored in remaining time feature

        for i in range(0,10):
            del tabular_df['player{}_tScore'.format(i)]
            del tabular_df['player{}_ctScore'.format(i)]
            del tabular_df['player{}_endTScore'.format(i)]
            del tabular_df['player{}_endCTScore'.format(i)]

        return tabular_df
    

    # 9. Add numerical match id
    def __TABULAR_add_numerical_match_id__(self, tabular_df):

        if self.add_numerical_match_id:

            if self.numerical_match_id is None:
                raise ValueError("Numerical match id is not provided.")
            elif type(self.numerical_match_id) is not int:
                raise ValueError("Numerical match id must be an integer.")
            
            new_columns = pd.DataFrame({
                'numerical_match_id': self.numerical_match_id
            }, index=tabular_df.index)
            tabular_df = pd.concat([tabular_df, new_columns], axis=1)

            return tabular_df
        else:
            return tabular_df


    # 10. Split the bombsites by 3x3 matrix for bomb position feature
    def __EXT_get_bomb_mx_coordinate__(self, row):
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

    def __TABULAR_bombsite_3x3_matrix_split_for_bomb_pos_feature__(self, df):
            
        new_columns = pd.DataFrame({
            'bomb_mx_pos': 0
        }, index=df.index)

        df = pd.concat([df, new_columns], axis=1)
        
        df.loc[(df['is_bomb_planted_at_A_site'] == 1) | (df['is_bomb_planted_at_B_site'] == 1), 'bomb_mx_pos'] = df.apply(self.__EXT_get_bomb_mx_coordinate__, axis=1)

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
    


    # 11. Function to extend the dataframe with copies of the rounds with varied player permutations
    def __TABULAR_vary_player_permutation__(self, df, num_permutations_per_round=3):
        """
        Function to extend the dataframe with copies of the rounds with varied player permutations
        """

        # Get the unique rounds and store team 1 and two player numbers
        team_1_indicies = [0, 1, 2, 3, 4]
        team_2_indicies = [5, 6, 7, 8, 9]
        rounds = df['roundNum'].unique()

        for rnd in rounds:
            for _permutation in range(num_permutations_per_round):
                # Get the round dataframe
                round_df = df[df['roundNum'] == rnd].copy()
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



    # 12. Rearrange the player columns so that the CTs are always from 0 to 4 and Ts are from 5 to 9
    def __TABULAR_refactor_player_columns_to_CT_T__(self, df):

        # Separate the CT and T halves
        team_1_ct = df.loc[df['player0_isCT'] == True].copy()
        team_2_ct = df.loc[df['player0_isCT'] == False].copy()

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
            'tick',
            'roundNum',


            'CT0_name', 'CT0_x', 'CT0_y', 'CT0_z', 'CT0_eyeX', 'CT0_eyeY', 'CT0_eyeZ', 'CT0_velocityX',
                'CT0_velocityY', 'CT0_velocityZ', 'CT0_hp', 'CT0_armor', 'CT0_flashGrenades', 'CT0_smokeGrenades', 'CT0_heGrenades', 'CT0_totalUtility', 'CT0_isAlive',
                'CT0_isReloading', 'CT0_isBlinded', 'CT0_isDucking', 'CT0_isDefusing', 'CT0_isPlanting', 'CT0_isUnknown', 'CT0_isScoped', 'CT0_equipmentValue',
                'CT0_equipmentValueRoundStart', 'CT0_hasHelmet', 'CT0_hasDefuse', 'CT0_hasBomb', 'CT0_isCT', 'CT0_stat_kills', 'CT0_stat_HSK', 'CT0_stat_openKills',
                'CT0_stat_tradeKills', 'CT0_stat_deaths', 'CT0_stat_openDeaths', 'CT0_stat_assists', 'CT0_stat_flashAssists', 'CT0_stat_damage', 'CT0_stat_weaponDamage',
                'CT0_stat_nadeDamage', 'CT0_activeWeapon_C4', 'CT0_activeWeapon_Knife', 'CT0_activeWeapon_Taser', 'CT0_activeWeapon_USP-S', 'CT0_activeWeapon_P2000', 'CT0_activeWeapon_Glock-18',
                'CT0_activeWeapon_Dual Berettas', 'CT0_activeWeapon_P250', 'CT0_activeWeapon_Tec-9', 'CT0_activeWeapon_CZ75 Auto', 'CT0_activeWeapon_Five-SeveN', 'CT0_activeWeapon_Desert Eagle', 'CT0_activeWeapon_MAC-10', 'CT0_activeWeapon_MP9',
                'CT0_activeWeapon_MP7', 'CT0_activeWeapon_MP5-SD', 'CT0_activeWeapon_UMP-45', 'CT0_activeWeapon_PP-Bizon', 'CT0_activeWeapon_P90', 'CT0_activeWeapon_Nova', 'CT0_activeWeapon_XM1014', 'CT0_activeWeapon_Sawed-Off', 'CT0_activeWeapon_MAG-7',
                'CT0_activeWeapon_M249', 'CT0_activeWeapon_Negev', 'CT0_activeWeapon_FAMAS', 'CT0_activeWeapon_Galil AR', 'CT0_activeWeapon_AK-47', 'CT0_activeWeapon_M4A4', 'CT0_activeWeapon_M4A1', 'CT0_activeWeapon_SG 553', 'CT0_activeWeapon_AUG',
                'CT0_activeWeapon_SSG 08', 'CT0_activeWeapon_AWP', 'CT0_activeWeapon_G3SG1', 'CT0_activeWeapon_SCAR-20', 'CT0_activeWeapon_HE Grenade', 'CT0_activeWeapon_Flashbang', 'CT0_activeWeapon_Smoke Grenade', 'CT0_activeWeapon_Incendiary Grenade',
                'CT0_activeWeapon_Molotov', 'CT0_activeWeapon_Decoy Grenade', 'CT0_overall_rating_2.0', 'CT0_overall_DPR', 'CT0_overall_KAST', 'CT0_overall_Impact', 'CT0_overall_ADR', 'CT0_overall_KPR', 'CT0_overall_total_kills', 'CT0_overall_HS%',
                'CT0_overall_total_deaths', 'CT0_overall_KD_ratio', 'CT0_overall_dmgPR', 'CT0_overall_grenade_dmgPR', 'CT0_overall_maps_played', 'CT0_overall_saved_by_teammatePR', 'CT0_overall_saved_teammatesPR', 'CT0_overall_opening_kill_rating', 'CT0_overall_team_W%_after_opening',
                'CT0_overall_opening_kill_in_W_rounds', 'CT0_overall_rating_1.0_all_Career', 'CT0_overall_clutches_1on1_ratio', 'CT0_overall_clutches_won_1on1', 'CT0_overall_clutches_won_1on2', 'CT0_overall_clutches_won_1on3', 'CT0_overall_clutches_won_1on4', 'CT0_overall_clutches_won_1on5', 
            'CT1_name', 'CT1_x', 'CT1_y', 'CT1_z', 'CT1_eyeX', 'CT1_eyeY', 'CT1_eyeZ', 'CT1_velocityX',
                'CT1_velocityY', 'CT1_velocityZ', 'CT1_hp', 'CT1_armor', 'CT1_flashGrenades', 'CT1_smokeGrenades', 'CT1_heGrenades', 'CT1_totalUtility', 'CT1_isAlive',
                'CT1_isReloading', 'CT1_isBlinded', 'CT1_isDucking', 'CT1_isDefusing', 'CT1_isPlanting', 'CT1_isUnknown', 'CT1_isScoped', 'CT1_equipmentValue',
                'CT1_equipmentValueRoundStart', 'CT1_hasHelmet', 'CT1_hasDefuse', 'CT1_hasBomb', 'CT1_isCT', 'CT1_stat_kills', 'CT1_stat_HSK', 'CT1_stat_openKills',
                'CT1_stat_tradeKills', 'CT1_stat_deaths', 'CT1_stat_openDeaths', 'CT1_stat_assists', 'CT1_stat_flashAssists', 'CT1_stat_damage', 'CT1_stat_weaponDamage',
                'CT1_stat_nadeDamage', 'CT1_activeWeapon_C4', 'CT1_activeWeapon_Knife', 'CT1_activeWeapon_Taser', 'CT1_activeWeapon_USP-S', 'CT1_activeWeapon_P2000', 'CT1_activeWeapon_Glock-18',
                'CT1_activeWeapon_Dual Berettas', 'CT1_activeWeapon_P250', 'CT1_activeWeapon_Tec-9', 'CT1_activeWeapon_CZ75 Auto', 'CT1_activeWeapon_Five-SeveN', 'CT1_activeWeapon_Desert Eagle', 'CT1_activeWeapon_MAC-10', 'CT1_activeWeapon_MP9',
                'CT1_activeWeapon_MP7', 'CT1_activeWeapon_MP5-SD', 'CT1_activeWeapon_UMP-45', 'CT1_activeWeapon_PP-Bizon', 'CT1_activeWeapon_P90', 'CT1_activeWeapon_Nova', 'CT1_activeWeapon_XM1014', 'CT1_activeWeapon_Sawed-Off', 'CT1_activeWeapon_MAG-7',
                'CT1_activeWeapon_M249', 'CT1_activeWeapon_Negev', 'CT1_activeWeapon_FAMAS', 'CT1_activeWeapon_Galil AR', 'CT1_activeWeapon_AK-47', 'CT1_activeWeapon_M4A4', 'CT1_activeWeapon_M4A1', 'CT1_activeWeapon_SG 553', 'CT1_activeWeapon_AUG',
                'CT1_activeWeapon_SSG 08', 'CT1_activeWeapon_AWP', 'CT1_activeWeapon_G3SG1', 'CT1_activeWeapon_SCAR-20', 'CT1_activeWeapon_HE Grenade', 'CT1_activeWeapon_Flashbang', 'CT1_activeWeapon_Smoke Grenade', 'CT1_activeWeapon_Incendiary Grenade',
                'CT1_activeWeapon_Molotov', 'CT1_activeWeapon_Decoy Grenade', 'CT1_overall_rating_2.0', 'CT1_overall_DPR', 'CT1_overall_KAST', 'CT1_overall_Impact', 'CT1_overall_ADR', 'CT1_overall_KPR', 'CT1_overall_total_kills', 'CT1_overall_HS%',
                'CT1_overall_total_deaths', 'CT1_overall_KD_ratio', 'CT1_overall_dmgPR', 'CT1_overall_grenade_dmgPR', 'CT1_overall_maps_played', 'CT1_overall_saved_by_teammatePR', 'CT1_overall_saved_teammatesPR', 'CT1_overall_opening_kill_rating', 'CT1_overall_team_W%_after_opening',
                'CT1_overall_opening_kill_in_W_rounds', 'CT1_overall_rating_1.0_all_Career', 'CT1_overall_clutches_1on1_ratio', 'CT1_overall_clutches_won_1on1', 'CT1_overall_clutches_won_1on2', 'CT1_overall_clutches_won_1on3', 'CT1_overall_clutches_won_1on4', 'CT1_overall_clutches_won_1on5', 
            'CT2_name', 'CT2_x', 'CT2_y', 'CT2_z', 'CT2_eyeX', 'CT2_eyeY', 'CT2_eyeZ', 'CT2_velocityX',
                'CT2_velocityY', 'CT2_velocityZ', 'CT2_hp', 'CT2_armor', 'CT2_flashGrenades', 'CT2_smokeGrenades', 'CT2_heGrenades', 'CT2_totalUtility', 'CT2_isAlive',
                'CT2_isReloading', 'CT2_isBlinded', 'CT2_isDucking', 'CT2_isDefusing', 'CT2_isPlanting', 'CT2_isUnknown', 'CT2_isScoped', 'CT2_equipmentValue',
                'CT2_equipmentValueRoundStart', 'CT2_hasHelmet', 'CT2_hasDefuse', 'CT2_hasBomb', 'CT2_isCT', 'CT2_stat_kills', 'CT2_stat_HSK', 'CT2_stat_openKills',
                'CT2_stat_tradeKills', 'CT2_stat_deaths', 'CT2_stat_openDeaths', 'CT2_stat_assists', 'CT2_stat_flashAssists', 'CT2_stat_damage', 'CT2_stat_weaponDamage',
                'CT2_stat_nadeDamage', 'CT2_activeWeapon_C4', 'CT2_activeWeapon_Knife', 'CT2_activeWeapon_Taser', 'CT2_activeWeapon_USP-S', 'CT2_activeWeapon_P2000', 'CT2_activeWeapon_Glock-18',
                'CT2_activeWeapon_Dual Berettas', 'CT2_activeWeapon_P250', 'CT2_activeWeapon_Tec-9', 'CT2_activeWeapon_CZ75 Auto', 'CT2_activeWeapon_Five-SeveN', 'CT2_activeWeapon_Desert Eagle', 'CT2_activeWeapon_MAC-10', 'CT2_activeWeapon_MP9',
                'CT2_activeWeapon_MP7', 'CT2_activeWeapon_MP5-SD', 'CT2_activeWeapon_UMP-45', 'CT2_activeWeapon_PP-Bizon', 'CT2_activeWeapon_P90', 'CT2_activeWeapon_Nova', 'CT2_activeWeapon_XM1014', 'CT2_activeWeapon_Sawed-Off', 'CT2_activeWeapon_MAG-7',
                'CT2_activeWeapon_M249', 'CT2_activeWeapon_Negev', 'CT2_activeWeapon_FAMAS', 'CT2_activeWeapon_Galil AR', 'CT2_activeWeapon_AK-47', 'CT2_activeWeapon_M4A4', 'CT2_activeWeapon_M4A1', 'CT2_activeWeapon_SG 553', 'CT2_activeWeapon_AUG',
                'CT2_activeWeapon_SSG 08', 'CT2_activeWeapon_AWP', 'CT2_activeWeapon_G3SG1', 'CT2_activeWeapon_SCAR-20', 'CT2_activeWeapon_HE Grenade', 'CT2_activeWeapon_Flashbang', 'CT2_activeWeapon_Smoke Grenade', 'CT2_activeWeapon_Incendiary Grenade',
                'CT2_activeWeapon_Molotov', 'CT2_activeWeapon_Decoy Grenade', 'CT2_overall_rating_2.0', 'CT2_overall_DPR', 'CT2_overall_KAST', 'CT2_overall_Impact', 'CT2_overall_ADR', 'CT2_overall_KPR', 'CT2_overall_total_kills', 'CT2_overall_HS%',
                'CT2_overall_total_deaths', 'CT2_overall_KD_ratio', 'CT2_overall_dmgPR', 'CT2_overall_grenade_dmgPR', 'CT2_overall_maps_played', 'CT2_overall_saved_by_teammatePR', 'CT2_overall_saved_teammatesPR', 'CT2_overall_opening_kill_rating', 'CT2_overall_team_W%_after_opening',
                'CT2_overall_opening_kill_in_W_rounds', 'CT2_overall_rating_1.0_all_Career', 'CT2_overall_clutches_1on1_ratio', 'CT2_overall_clutches_won_1on1', 'CT2_overall_clutches_won_1on2', 'CT2_overall_clutches_won_1on3', 'CT2_overall_clutches_won_1on4', 'CT2_overall_clutches_won_1on5', 
            'CT3_name', 'CT3_x', 'CT3_y', 'CT3_z', 'CT3_eyeX', 'CT3_eyeY', 'CT3_eyeZ', 'CT3_velocityX',
                'CT3_velocityY', 'CT3_velocityZ', 'CT3_hp', 'CT3_armor', 'CT3_flashGrenades', 'CT3_smokeGrenades', 'CT3_heGrenades', 'CT3_totalUtility', 'CT3_isAlive',
                'CT3_isReloading', 'CT3_isBlinded', 'CT3_isDucking', 'CT3_isDefusing', 'CT3_isPlanting', 'CT3_isUnknown', 'CT3_isScoped', 'CT3_equipmentValue',
                'CT3_equipmentValueRoundStart', 'CT3_hasHelmet', 'CT3_hasDefuse', 'CT3_hasBomb', 'CT3_isCT', 'CT3_stat_kills', 'CT3_stat_HSK', 'CT3_stat_openKills',
                'CT3_stat_tradeKills', 'CT3_stat_deaths', 'CT3_stat_openDeaths', 'CT3_stat_assists', 'CT3_stat_flashAssists', 'CT3_stat_damage', 'CT3_stat_weaponDamage',
                'CT3_stat_nadeDamage', 'CT3_activeWeapon_C4', 'CT3_activeWeapon_Knife', 'CT3_activeWeapon_Taser', 'CT3_activeWeapon_USP-S', 'CT3_activeWeapon_P2000', 'CT3_activeWeapon_Glock-18',
                'CT3_activeWeapon_Dual Berettas', 'CT3_activeWeapon_P250', 'CT3_activeWeapon_Tec-9', 'CT3_activeWeapon_CZ75 Auto', 'CT3_activeWeapon_Five-SeveN', 'CT3_activeWeapon_Desert Eagle', 'CT3_activeWeapon_MAC-10', 'CT3_activeWeapon_MP9',
                'CT3_activeWeapon_MP7', 'CT3_activeWeapon_MP5-SD', 'CT3_activeWeapon_UMP-45', 'CT3_activeWeapon_PP-Bizon', 'CT3_activeWeapon_P90', 'CT3_activeWeapon_Nova', 'CT3_activeWeapon_XM1014', 'CT3_activeWeapon_Sawed-Off', 'CT3_activeWeapon_MAG-7',
                'CT3_activeWeapon_M249', 'CT3_activeWeapon_Negev', 'CT3_activeWeapon_FAMAS', 'CT3_activeWeapon_Galil AR', 'CT3_activeWeapon_AK-47', 'CT3_activeWeapon_M4A4', 'CT3_activeWeapon_M4A1', 'CT3_activeWeapon_SG 553', 'CT3_activeWeapon_AUG',
                'CT3_activeWeapon_SSG 08', 'CT3_activeWeapon_AWP', 'CT3_activeWeapon_G3SG1', 'CT3_activeWeapon_SCAR-20', 'CT3_activeWeapon_HE Grenade', 'CT3_activeWeapon_Flashbang', 'CT3_activeWeapon_Smoke Grenade', 'CT3_activeWeapon_Incendiary Grenade',
                'CT3_activeWeapon_Molotov', 'CT3_activeWeapon_Decoy Grenade', 'CT3_overall_rating_2.0', 'CT3_overall_DPR', 'CT3_overall_KAST', 'CT3_overall_Impact', 'CT3_overall_ADR', 'CT3_overall_KPR', 'CT3_overall_total_kills', 'CT3_overall_HS%',
                'CT3_overall_total_deaths', 'CT3_overall_KD_ratio', 'CT3_overall_dmgPR', 'CT3_overall_grenade_dmgPR', 'CT3_overall_maps_played', 'CT3_overall_saved_by_teammatePR', 'CT3_overall_saved_teammatesPR', 'CT3_overall_opening_kill_rating', 'CT3_overall_team_W%_after_opening',
                'CT3_overall_opening_kill_in_W_rounds', 'CT3_overall_rating_1.0_all_Career', 'CT3_overall_clutches_1on1_ratio', 'CT3_overall_clutches_won_1on1', 'CT3_overall_clutches_won_1on2', 'CT3_overall_clutches_won_1on3', 'CT3_overall_clutches_won_1on4', 'CT3_overall_clutches_won_1on5', 
            'CT4_name', 'CT4_x', 'CT4_y', 'CT4_z', 'CT4_eyeX', 'CT4_eyeY', 'CT4_eyeZ', 'CT4_velocityX',
                'CT4_velocityY', 'CT4_velocityZ', 'CT4_hp', 'CT4_armor', 'CT4_flashGrenades', 'CT4_smokeGrenades', 'CT4_heGrenades', 'CT4_totalUtility', 'CT4_isAlive',
                'CT4_isReloading', 'CT4_isBlinded', 'CT4_isDucking', 'CT4_isDefusing', 'CT4_isPlanting', 'CT4_isUnknown', 'CT4_isScoped', 'CT4_equipmentValue',
                'CT4_equipmentValueRoundStart', 'CT4_hasHelmet', 'CT4_hasDefuse', 'CT4_hasBomb', 'CT4_isCT', 'CT4_stat_kills', 'CT4_stat_HSK', 'CT4_stat_openKills',
                'CT4_stat_tradeKills', 'CT4_stat_deaths', 'CT4_stat_openDeaths', 'CT4_stat_assists', 'CT4_stat_flashAssists', 'CT4_stat_damage', 'CT4_stat_weaponDamage',
                'CT4_stat_nadeDamage', 'CT4_activeWeapon_C4', 'CT4_activeWeapon_Knife', 'CT4_activeWeapon_Taser', 'CT4_activeWeapon_USP-S', 'CT4_activeWeapon_P2000', 'CT4_activeWeapon_Glock-18',
                'CT4_activeWeapon_Dual Berettas', 'CT4_activeWeapon_P250', 'CT4_activeWeapon_Tec-9', 'CT4_activeWeapon_CZ75 Auto', 'CT4_activeWeapon_Five-SeveN', 'CT4_activeWeapon_Desert Eagle', 'CT4_activeWeapon_MAC-10', 'CT4_activeWeapon_MP9',
                'CT4_activeWeapon_MP7', 'CT4_activeWeapon_MP5-SD', 'CT4_activeWeapon_UMP-45', 'CT4_activeWeapon_PP-Bizon', 'CT4_activeWeapon_P90', 'CT4_activeWeapon_Nova', 'CT4_activeWeapon_XM1014', 'CT4_activeWeapon_Sawed-Off', 'CT4_activeWeapon_MAG-7',
                'CT4_activeWeapon_M249', 'CT4_activeWeapon_Negev', 'CT4_activeWeapon_FAMAS', 'CT4_activeWeapon_Galil AR', 'CT4_activeWeapon_AK-47', 'CT4_activeWeapon_M4A4', 'CT4_activeWeapon_M4A1', 'CT4_activeWeapon_SG 553', 'CT4_activeWeapon_AUG',
                'CT4_activeWeapon_SSG 08', 'CT4_activeWeapon_AWP', 'CT4_activeWeapon_G3SG1', 'CT4_activeWeapon_SCAR-20', 'CT4_activeWeapon_HE Grenade', 'CT4_activeWeapon_Flashbang', 'CT4_activeWeapon_Smoke Grenade', 'CT4_activeWeapon_Incendiary Grenade',
                'CT4_activeWeapon_Molotov', 'CT4_activeWeapon_Decoy Grenade', 'CT4_overall_rating_2.0', 'CT4_overall_DPR', 'CT4_overall_KAST', 'CT4_overall_Impact', 'CT4_overall_ADR', 'CT4_overall_KPR', 'CT4_overall_total_kills', 'CT4_overall_HS%',
                'CT4_overall_total_deaths', 'CT4_overall_KD_ratio', 'CT4_overall_dmgPR', 'CT4_overall_grenade_dmgPR', 'CT4_overall_maps_played', 'CT4_overall_saved_by_teammatePR', 'CT4_overall_saved_teammatesPR', 'CT4_overall_opening_kill_rating', 'CT4_overall_team_W%_after_opening',
                'CT4_overall_opening_kill_in_W_rounds', 'CT4_overall_rating_1.0_all_Career', 'CT4_overall_clutches_1on1_ratio', 'CT4_overall_clutches_won_1on1', 'CT4_overall_clutches_won_1on2', 'CT4_overall_clutches_won_1on3', 'CT4_overall_clutches_won_1on4', 'CT4_overall_clutches_won_1on5', 

            'T5_name', 'T5_x', 'T5_y', 'T5_z', 'T5_eyeX', 'T5_eyeY', 'T5_eyeZ', 'T5_velocityX',
                'T5_velocityY', 'T5_velocityZ', 'T5_hp', 'T5_armor', 'T5_flashGrenades', 'T5_smokeGrenades', 'T5_heGrenades', 'T5_totalUtility', 'T5_isAlive',
                'T5_isReloading', 'T5_isBlinded', 'T5_isDucking', 'T5_isDefusing', 'T5_isPlanting', 'T5_isUnknown', 'T5_isScoped', 'T5_equipmentValue',
                'T5_equipmentValueRoundStart', 'T5_hasHelmet', 'T5_hasDefuse', 'T5_hasBomb', 'T5_isCT', 'T5_stat_kills', 'T5_stat_HSK', 'T5_stat_openKills',
                'T5_stat_tradeKills', 'T5_stat_deaths', 'T5_stat_openDeaths', 'T5_stat_assists', 'T5_stat_flashAssists', 'T5_stat_damage', 'T5_stat_weaponDamage',
                'T5_stat_nadeDamage', 'T5_activeWeapon_C4', 'T5_activeWeapon_Knife', 'T5_activeWeapon_Taser', 'T5_activeWeapon_USP-S', 'T5_activeWeapon_P2000', 'T5_activeWeapon_Glock-18',
                'T5_activeWeapon_Dual Berettas', 'T5_activeWeapon_P250', 'T5_activeWeapon_Tec-9', 'T5_activeWeapon_CZ75 Auto', 'T5_activeWeapon_Five-SeveN', 'T5_activeWeapon_Desert Eagle', 'T5_activeWeapon_MAC-10', 'T5_activeWeapon_MP9',
                'T5_activeWeapon_MP7', 'T5_activeWeapon_MP5-SD', 'T5_activeWeapon_UMP-45', 'T5_activeWeapon_PP-Bizon', 'T5_activeWeapon_P90', 'T5_activeWeapon_Nova', 'T5_activeWeapon_XM1014', 'T5_activeWeapon_Sawed-Off', 'T5_activeWeapon_MAG-7',
                'T5_activeWeapon_M249', 'T5_activeWeapon_Negev', 'T5_activeWeapon_FAMAS', 'T5_activeWeapon_Galil AR', 'T5_activeWeapon_AK-47', 'T5_activeWeapon_M4A4', 'T5_activeWeapon_M4A1', 'T5_activeWeapon_SG 553', 'T5_activeWeapon_AUG',
                'T5_activeWeapon_SSG 08', 'T5_activeWeapon_AWP', 'T5_activeWeapon_G3SG1', 'T5_activeWeapon_SCAR-20', 'T5_activeWeapon_HE Grenade', 'T5_activeWeapon_Flashbang', 'T5_activeWeapon_Smoke Grenade', 'T5_activeWeapon_Incendiary Grenade',
                'T5_activeWeapon_Molotov', 'T5_activeWeapon_Decoy Grenade', 'T5_overall_rating_2.0', 'T5_overall_DPR', 'T5_overall_KAST', 'T5_overall_Impact', 'T5_overall_ADR', 'T5_overall_KPR', 'T5_overall_total_kills', 'T5_overall_HS%',
                'T5_overall_total_deaths', 'T5_overall_KD_ratio', 'T5_overall_dmgPR', 'T5_overall_grenade_dmgPR', 'T5_overall_maps_played', 'T5_overall_saved_by_teammatePR', 'T5_overall_saved_teammatesPR', 'T5_overall_opening_kill_rating', 'T5_overall_team_W%_after_opening',
                'T5_overall_opening_kill_in_W_rounds', 'T5_overall_rating_1.0_all_Career', 'T5_overall_clutches_1on1_ratio', 'T5_overall_clutches_won_1on1', 'T5_overall_clutches_won_1on2', 'T5_overall_clutches_won_1on3', 'T5_overall_clutches_won_1on4', 'T5_overall_clutches_won_1on5', 
            'T6_name', 'T6_x', 'T6_y', 'T6_z', 'T6_eyeX', 'T6_eyeY', 'T6_eyeZ', 'T6_velocityX',
                'T6_velocityY', 'T6_velocityZ', 'T6_hp', 'T6_armor', 'T6_flashGrenades', 'T6_smokeGrenades', 'T6_heGrenades', 'T6_totalUtility', 'T6_isAlive',
                'T6_isReloading', 'T6_isBlinded', 'T6_isDucking', 'T6_isDefusing', 'T6_isPlanting', 'T6_isUnknown', 'T6_isScoped', 'T6_equipmentValue',
                'T6_equipmentValueRoundStart', 'T6_hasHelmet', 'T6_hasDefuse', 'T6_hasBomb', 'T6_isCT', 'T6_stat_kills', 'T6_stat_HSK', 'T6_stat_openKills',
                'T6_stat_tradeKills', 'T6_stat_deaths', 'T6_stat_openDeaths', 'T6_stat_assists', 'T6_stat_flashAssists', 'T6_stat_damage', 'T6_stat_weaponDamage',
                'T6_stat_nadeDamage', 'T6_activeWeapon_C4', 'T6_activeWeapon_Knife', 'T6_activeWeapon_Taser', 'T6_activeWeapon_USP-S', 'T6_activeWeapon_P2000', 'T6_activeWeapon_Glock-18',
                'T6_activeWeapon_Dual Berettas', 'T6_activeWeapon_P250', 'T6_activeWeapon_Tec-9', 'T6_activeWeapon_CZ75 Auto', 'T6_activeWeapon_Five-SeveN', 'T6_activeWeapon_Desert Eagle', 'T6_activeWeapon_MAC-10', 'T6_activeWeapon_MP9',
                'T6_activeWeapon_MP7', 'T6_activeWeapon_MP5-SD', 'T6_activeWeapon_UMP-45', 'T6_activeWeapon_PP-Bizon', 'T6_activeWeapon_P90', 'T6_activeWeapon_Nova', 'T6_activeWeapon_XM1014', 'T6_activeWeapon_Sawed-Off', 'T6_activeWeapon_MAG-7',
                'T6_activeWeapon_M249', 'T6_activeWeapon_Negev', 'T6_activeWeapon_FAMAS', 'T6_activeWeapon_Galil AR', 'T6_activeWeapon_AK-47', 'T6_activeWeapon_M4A4', 'T6_activeWeapon_M4A1', 'T6_activeWeapon_SG 553', 'T6_activeWeapon_AUG',
                'T6_activeWeapon_SSG 08', 'T6_activeWeapon_AWP', 'T6_activeWeapon_G3SG1', 'T6_activeWeapon_SCAR-20', 'T6_activeWeapon_HE Grenade', 'T6_activeWeapon_Flashbang', 'T6_activeWeapon_Smoke Grenade', 'T6_activeWeapon_Incendiary Grenade',
                'T6_activeWeapon_Molotov', 'T6_activeWeapon_Decoy Grenade', 'T6_overall_rating_2.0', 'T6_overall_DPR', 'T6_overall_KAST', 'T6_overall_Impact', 'T6_overall_ADR', 'T6_overall_KPR', 'T6_overall_total_kills', 'T6_overall_HS%',
                'T6_overall_total_deaths', 'T6_overall_KD_ratio', 'T6_overall_dmgPR', 'T6_overall_grenade_dmgPR', 'T6_overall_maps_played', 'T6_overall_saved_by_teammatePR', 'T6_overall_saved_teammatesPR', 'T6_overall_opening_kill_rating', 'T6_overall_team_W%_after_opening',
                'T6_overall_opening_kill_in_W_rounds', 'T6_overall_rating_1.0_all_Career', 'T6_overall_clutches_1on1_ratio', 'T6_overall_clutches_won_1on1', 'T6_overall_clutches_won_1on2', 'T6_overall_clutches_won_1on3', 'T6_overall_clutches_won_1on4', 'T6_overall_clutches_won_1on5', 
            'T7_name', 'T7_x', 'T7_y', 'T7_z', 'T7_eyeX', 'T7_eyeY', 'T7_eyeZ', 'T7_velocityX',
                'T7_velocityY', 'T7_velocityZ', 'T7_hp', 'T7_armor', 'T7_flashGrenades', 'T7_smokeGrenades', 'T7_heGrenades', 'T7_totalUtility', 'T7_isAlive',
                'T7_isReloading', 'T7_isBlinded', 'T7_isDucking', 'T7_isDefusing', 'T7_isPlanting', 'T7_isUnknown', 'T7_isScoped', 'T7_equipmentValue',
                'T7_equipmentValueRoundStart', 'T7_hasHelmet', 'T7_hasDefuse', 'T7_hasBomb', 'T7_isCT', 'T7_stat_kills', 'T7_stat_HSK', 'T7_stat_openKills',
                'T7_stat_tradeKills', 'T7_stat_deaths', 'T7_stat_openDeaths', 'T7_stat_assists', 'T7_stat_flashAssists', 'T7_stat_damage', 'T7_stat_weaponDamage',
                'T7_stat_nadeDamage', 'T7_activeWeapon_C4', 'T7_activeWeapon_Knife', 'T7_activeWeapon_Taser', 'T7_activeWeapon_USP-S', 'T7_activeWeapon_P2000', 'T7_activeWeapon_Glock-18',
                'T7_activeWeapon_Dual Berettas', 'T7_activeWeapon_P250', 'T7_activeWeapon_Tec-9', 'T7_activeWeapon_CZ75 Auto', 'T7_activeWeapon_Five-SeveN', 'T7_activeWeapon_Desert Eagle', 'T7_activeWeapon_MAC-10', 'T7_activeWeapon_MP9',
                'T7_activeWeapon_MP7', 'T7_activeWeapon_MP5-SD', 'T7_activeWeapon_UMP-45', 'T7_activeWeapon_PP-Bizon', 'T7_activeWeapon_P90', 'T7_activeWeapon_Nova', 'T7_activeWeapon_XM1014', 'T7_activeWeapon_Sawed-Off', 'T7_activeWeapon_MAG-7',
                'T7_activeWeapon_M249', 'T7_activeWeapon_Negev', 'T7_activeWeapon_FAMAS', 'T7_activeWeapon_Galil AR', 'T7_activeWeapon_AK-47', 'T7_activeWeapon_M4A4', 'T7_activeWeapon_M4A1', 'T7_activeWeapon_SG 553', 'T7_activeWeapon_AUG',
                'T7_activeWeapon_SSG 08', 'T7_activeWeapon_AWP', 'T7_activeWeapon_G3SG1', 'T7_activeWeapon_SCAR-20', 'T7_activeWeapon_HE Grenade', 'T7_activeWeapon_Flashbang', 'T7_activeWeapon_Smoke Grenade', 'T7_activeWeapon_Incendiary Grenade',
                'T7_activeWeapon_Molotov', 'T7_activeWeapon_Decoy Grenade', 'T7_overall_rating_2.0', 'T7_overall_DPR', 'T7_overall_KAST', 'T7_overall_Impact', 'T7_overall_ADR', 'T7_overall_KPR', 'T7_overall_total_kills', 'T7_overall_HS%',
                'T7_overall_total_deaths', 'T7_overall_KD_ratio', 'T7_overall_dmgPR', 'T7_overall_grenade_dmgPR', 'T7_overall_maps_played', 'T7_overall_saved_by_teammatePR', 'T7_overall_saved_teammatesPR', 'T7_overall_opening_kill_rating', 'T7_overall_team_W%_after_opening',
                'T7_overall_opening_kill_in_W_rounds', 'T7_overall_rating_1.0_all_Career', 'T7_overall_clutches_1on1_ratio', 'T7_overall_clutches_won_1on1', 'T7_overall_clutches_won_1on2', 'T7_overall_clutches_won_1on3', 'T7_overall_clutches_won_1on4', 'T7_overall_clutches_won_1on5', 
            'T8_name', 'T8_x', 'T8_y', 'T8_z', 'T8_eyeX', 'T8_eyeY', 'T8_eyeZ', 'T8_velocityX',
                'T8_velocityY', 'T8_velocityZ', 'T8_hp', 'T8_armor', 'T8_flashGrenades', 'T8_smokeGrenades', 'T8_heGrenades', 'T8_totalUtility', 'T8_isAlive',
                'T8_isReloading', 'T8_isBlinded', 'T8_isDucking', 'T8_isDefusing', 'T8_isPlanting', 'T8_isUnknown', 'T8_isScoped', 'T8_equipmentValue',
                'T8_equipmentValueRoundStart', 'T8_hasHelmet', 'T8_hasDefuse', 'T8_hasBomb', 'T8_isCT', 'T8_stat_kills', 'T8_stat_HSK', 'T8_stat_openKills',
                'T8_stat_tradeKills', 'T8_stat_deaths', 'T8_stat_openDeaths', 'T8_stat_assists', 'T8_stat_flashAssists', 'T8_stat_damage', 'T8_stat_weaponDamage',
                'T8_stat_nadeDamage', 'T8_activeWeapon_C4', 'T8_activeWeapon_Knife', 'T8_activeWeapon_Taser', 'T8_activeWeapon_USP-S', 'T8_activeWeapon_P2000', 'T8_activeWeapon_Glock-18',
                'T8_activeWeapon_Dual Berettas', 'T8_activeWeapon_P250', 'T8_activeWeapon_Tec-9', 'T8_activeWeapon_CZ75 Auto', 'T8_activeWeapon_Five-SeveN', 'T8_activeWeapon_Desert Eagle', 'T8_activeWeapon_MAC-10', 'T8_activeWeapon_MP9',
                'T8_activeWeapon_MP7', 'T8_activeWeapon_MP5-SD', 'T8_activeWeapon_UMP-45', 'T8_activeWeapon_PP-Bizon', 'T8_activeWeapon_P90', 'T8_activeWeapon_Nova', 'T8_activeWeapon_XM1014', 'T8_activeWeapon_Sawed-Off', 'T8_activeWeapon_MAG-7',
                'T8_activeWeapon_M249', 'T8_activeWeapon_Negev', 'T8_activeWeapon_FAMAS', 'T8_activeWeapon_Galil AR', 'T8_activeWeapon_AK-47', 'T8_activeWeapon_M4A4', 'T8_activeWeapon_M4A1', 'T8_activeWeapon_SG 553', 'T8_activeWeapon_AUG',
                'T8_activeWeapon_SSG 08', 'T8_activeWeapon_AWP', 'T8_activeWeapon_G3SG1', 'T8_activeWeapon_SCAR-20', 'T8_activeWeapon_HE Grenade', 'T8_activeWeapon_Flashbang', 'T8_activeWeapon_Smoke Grenade', 'T8_activeWeapon_Incendiary Grenade',
                'T8_activeWeapon_Molotov', 'T8_activeWeapon_Decoy Grenade', 'T8_overall_rating_2.0', 'T8_overall_DPR', 'T8_overall_KAST', 'T8_overall_Impact', 'T8_overall_ADR', 'T8_overall_KPR', 'T8_overall_total_kills', 'T8_overall_HS%',
                'T8_overall_total_deaths', 'T8_overall_KD_ratio', 'T8_overall_dmgPR', 'T8_overall_grenade_dmgPR', 'T8_overall_maps_played', 'T8_overall_saved_by_teammatePR', 'T8_overall_saved_teammatesPR', 'T8_overall_opening_kill_rating', 'T8_overall_team_W%_after_opening',
                'T8_overall_opening_kill_in_W_rounds', 'T8_overall_rating_1.0_all_Career', 'T8_overall_clutches_1on1_ratio', 'T8_overall_clutches_won_1on1', 'T8_overall_clutches_won_1on2', 'T8_overall_clutches_won_1on3', 'T8_overall_clutches_won_1on4', 'T8_overall_clutches_won_1on5', 
            'T9_name', 'T9_x', 'T9_y', 'T9_z', 'T9_eyeX', 'T9_eyeY', 'T9_eyeZ', 'T9_velocityX',
                'T9_velocityY', 'T9_velocityZ', 'T9_hp', 'T9_armor', 'T9_flashGrenades', 'T9_smokeGrenades', 'T9_heGrenades', 'T9_totalUtility', 'T9_isAlive',
                'T9_isReloading', 'T9_isBlinded', 'T9_isDucking', 'T9_isDefusing', 'T9_isPlanting', 'T9_isUnknown', 'T9_isScoped', 'T9_equipmentValue',
                'T9_equipmentValueRoundStart', 'T9_hasHelmet', 'T9_hasDefuse', 'T9_hasBomb', 'T9_isCT', 'T9_stat_kills', 'T9_stat_HSK', 'T9_stat_openKills',
                'T9_stat_tradeKills', 'T9_stat_deaths', 'T9_stat_openDeaths', 'T9_stat_assists', 'T9_stat_flashAssists', 'T9_stat_damage', 'T9_stat_weaponDamage',
                'T9_stat_nadeDamage', 'T9_activeWeapon_C4', 'T9_activeWeapon_Knife', 'T9_activeWeapon_Taser', 'T9_activeWeapon_USP-S', 'T9_activeWeapon_P2000', 'T9_activeWeapon_Glock-18',
                'T9_activeWeapon_Dual Berettas', 'T9_activeWeapon_P250', 'T9_activeWeapon_Tec-9', 'T9_activeWeapon_CZ75 Auto', 'T9_activeWeapon_Five-SeveN', 'T9_activeWeapon_Desert Eagle', 'T9_activeWeapon_MAC-10', 'T9_activeWeapon_MP9',
                'T9_activeWeapon_MP7', 'T9_activeWeapon_MP5-SD', 'T9_activeWeapon_UMP-45', 'T9_activeWeapon_PP-Bizon', 'T9_activeWeapon_P90', 'T9_activeWeapon_Nova', 'T9_activeWeapon_XM1014', 'T9_activeWeapon_Sawed-Off', 'T9_activeWeapon_MAG-7',
                'T9_activeWeapon_M249', 'T9_activeWeapon_Negev', 'T9_activeWeapon_FAMAS', 'T9_activeWeapon_Galil AR', 'T9_activeWeapon_AK-47', 'T9_activeWeapon_M4A4', 'T9_activeWeapon_M4A1', 'T9_activeWeapon_SG 553', 'T9_activeWeapon_AUG',
                'T9_activeWeapon_SSG 08', 'T9_activeWeapon_AWP', 'T9_activeWeapon_G3SG1', 'T9_activeWeapon_SCAR-20', 'T9_activeWeapon_HE Grenade', 'T9_activeWeapon_Flashbang', 'T9_activeWeapon_Smoke Grenade', 'T9_activeWeapon_Incendiary Grenade',
                'T9_activeWeapon_Molotov', 'T9_activeWeapon_Decoy Grenade', 'T9_overall_rating_2.0', 'T9_overall_DPR', 'T9_overall_KAST', 'T9_overall_Impact', 'T9_overall_ADR', 'T9_overall_KPR', 'T9_overall_total_kills', 'T9_overall_HS%',
                'T9_overall_total_deaths', 'T9_overall_KD_ratio', 'T9_overall_dmgPR', 'T9_overall_grenade_dmgPR', 'T9_overall_maps_played', 'T9_overall_saved_by_teammatePR', 'T9_overall_saved_teammatesPR', 'T9_overall_opening_kill_rating', 'T9_overall_team_W%_after_opening',
                'T9_overall_opening_kill_in_W_rounds', 'T9_overall_rating_1.0_all_Career', 'T9_overall_clutches_1on1_ratio', 'T9_overall_clutches_won_1on1', 'T9_overall_clutches_won_1on2', 'T9_overall_clutches_won_1on3', 'T9_overall_clutches_won_1on4', 'T9_overall_clutches_won_1on5', 


            'tScore', 'ctScore', 'endTScore', 'endCTScore', 'CT_winsRound', 'CT_aliveNum', 'T_aliveNum', 
                'CT_equipmentValue', 'T_equipmentValue', 'CT_totalHP', 'T_totalHP',
                'match_id', 'is_bomb_being_planted', 'is_bomb_being_defused',
                'is_bomb_defused', 'is_bomb_planted_at_A_site',
                'is_bomb_planted_at_B_site', 'bomb_X', 'bomb_Y', 'bomb_Z',
                'time_remaining', 'numerical_match_id', 'bomb_mx_pos1', 'bomb_mx_pos2',
                'bomb_mx_pos3', 'bomb_mx_pos4', 'bomb_mx_pos5', 'bomb_mx_pos6',
                'bomb_mx_pos7', 'bomb_mx_pos8', 'bomb_mx_pos9'
        ]

        # Rearrange the column order
        team_1_ct = team_1_ct[col_order]
        team_2_ct = team_2_ct[col_order]

        # Concatenate the two dataframes
        renamed_df = pd.concat([team_1_ct, team_2_ct])

        return renamed_df
