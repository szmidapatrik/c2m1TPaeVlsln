import pandas as pd
import numpy as np



class TabularDataCreator:



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



    # --------------------------------------------------------------------------------------------

    def __init__(self):
        pass



    # --------------------------------------------------------------------------------------------

    def format_match_data(
        self,
        match_file_name: str,
        tabular_data_folder_path: str, 
        player_stats_data_path: str, 
        tick_number: int = 1,
        missing_player_stats_data_path: str = None,
        add_numerical_match_id: bool = False,
        numerical_match_id: int = None,
    ):
        """
        Formats the match data and creates the tabular game-snapshot dataset. Parameters:
            - match_file_name: name of the match file,
            - tabular_data_folder_path: folder path of the parsed data,
            - player_stats_data_path: path of the player statistics data,
            - output_folder_path: folder path of the output,
            - missing_player_stats_data_path (optional): path of the missing player statistics data,
            - tick_number (optional): parse tick rate.
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

        return tabular_df





    def create_missing_player_stats_data(self, player_stats_data_path):
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
        fictive_player_df = pd.concat([fictive_player_df]*1000, ignore_index=True)

        # Create a DataFrame with the fictive player repeated 1000 times and with random values
        for col in numerical_cols:
            fictive_player_df[col] = np.random.normal(dist_values[col][0], dist_values[col][1], size=1000)
            if col not in ['KD_ratio', 'KD_diff']:
                fictive_player_df[col] = fictive_player_df[col].abs()
            if col in ['total_deaths', 'maps_played', 'rounds_played', 'rounds_with_kils', 'KD_diff', 'total_opening_kills', 'total_opening_deaths', 
                       '0_kill_rounds', '1_kill_rounds', '2_kill_rounds', '3_kill_rounds', '4_kill_rounds', '5_kill_rounds',
                       'rifle_kills', 'sniper_kills', 'smg_kills', 'pistol_kills', 'grenade_kills', 'other_kills', 'rating_2.0_1+', 'rating_2.0_1+_streak', 
                       'clutches_won_1on1', 'clutches_lost_1on1', 'clutches_won_1on2', 'clutches_won_1on3', 'clutches_won_1on4', 'clutches_won_1on5']:
                fictive_player_df[col] = fictive_player_df[col].apply(lambda x: int(x))

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
                    mpdf.to_csv(self.MISSING_PLAYER_STATS_DATA_PATH, index=False)
                    players[idx] = self.__EXT_insert_columns_into_player_dataframes__(mpdf, players[idx])
            
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

        # Create a DataFrame with a single column for match_id
        match_id_df = pd.DataFrame({'match_id': str(match_id)}, index=graph_data.index)
        graph_data_concatenated = pd.concat([graph_data, match_id_df], axis=1)

        return graph_data_concatenated


    # 7. Add bomb information to the dataset
    def __TABULAR_add_bomb_info_to_dataset__(self, tabular_df, bombdf):

        tabular_df['is_bomb_being_planted'] = 0
        tabular_df['is_bomb_being_defused'] = 0
        tabular_df['is_bomb_defused'] = 0
        tabular_df['is_bomb_planted_at_A_site'] = 0
        tabular_df['is_bomb_planted_at_B_site'] = 0
        tabular_df['bomb_X'] = 0.0
        tabular_df['bomb_Y'] = 0.0
        tabular_df['bomb_Z'] = 0.0

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
        tabular_df['sec'] = (tabular_df['tick'] - tabular_df['roundStartTick']) / 128
        tabular_df['time_remaining'] = 115 - tabular_df['sec']

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
            
            tabular_df['numerical_match_id'] = self.numerical_match_id
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
            
        df['bomb_mx_pos'] = 0
        
        df.loc[(df['is_bomb_planted_at_A_site'] == 1) | (df['is_bomb_planted_at_B_site'] == 1), 'bomb_mx_pos'] = df.apply(self.__EXT_get_bomb_mx_coordinate__, axis=1)

        # Dummify the bomb_mx_pos column and drop the original column
        df['bomb_mx_pos1'] = 0
        df['bomb_mx_pos2'] = 0
        df['bomb_mx_pos3'] = 0
        df['bomb_mx_pos4'] = 0
        df['bomb_mx_pos5'] = 0
        df['bomb_mx_pos6'] = 0
        df['bomb_mx_pos7'] = 0
        df['bomb_mx_pos8'] = 0
        df['bomb_mx_pos9'] = 0

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