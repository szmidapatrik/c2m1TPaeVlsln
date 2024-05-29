import pandas as pd
import numpy as np



class TabularDataCreator:

    # INPUT
    # Folder path constants
    TABULAR_DATA_FOLDER_PATH = None
    PLAYER_STATS_DATA_FOLDER_PATH = None
    MISSING_PLAYER_STATS_DATA_FOLDER_PATH = None


    # OUTPUT
    # Folder path constants
    OUTPUT_FOLDER_PATH = None




    # --------------------------------------------------------------------------------------------

    def __init__(
            self, 
            tabular_data_folder_path: str, 
            player_stats_data_folder_path: str, 
            missing_player_stats_data_folder_path: str,
            output_folder_path: str
    ):
        """
        Parameters:
            - tabular_data_folder_path: str,
            - inferno_graph_model_folder_path: str,
            - player_stats_data_folder_path: str,
            - output_folder_path: str,
        """
        # INPUT
        self.TABULAR_DATA_FOLDER_PATH = tabular_data_folder_path
        self.PLAYER_STATS_DATA_FOLDER_PATH = player_stats_data_folder_path
        self.MISSING_PLAYER_STATS_DATA_FOLDER_PATH = missing_player_stats_data_folder_path

        # OUTPUT
        self.OUTPUT_FOLDER_PATH = output_folder_path





    # --------------------------------------------------------------------------------------------

    def __get_needed_dataframes__(self, filename):

        # Read dataframes
        playerFrames = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/playerFrames/' + filename)
        kills = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH +'/kills/' + filename)
        rounds = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH +'/rounds/' + filename)
        bombEvents = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/bombEvents/' + filename)
        damages = pd.read_csv(self.TABULAR_DATA_FOLDER_PATH + '/damages/' + filename)

        # Filter columns
        rounds = rounds[['roundNum', 'tScore', "ctScore" ,'endTScore', 'endCTScore']]
        pf = playerFrames[['tick', 'roundNum', 'seconds', 'side', 'name', 'x', 'y', 'z','eyeX', 'eyeY', 'eyeZ', 'velocityX', 'velocityY', 'velocityZ',
            'hp', 'armor', 'activeWeapon','flashGrenades', 'smokeGrenades', 'heGrenades', 'totalUtility', 'isAlive', 'isReloading', 'isBlinded', 'isDucking',
            'isDefusing', 'isPlanting', 'isUnknown', 'isScoped', 'equipmentValue', 'equipmentValueRoundStart', 'hasHelmet','hasDefuse', 'hasBomb']]
        
        return pf, kills, rounds, bombEvents, damages





    def __calculate_ingame_features_from_needed_dataframes__(self, pf, kills, rounds, damages):
    
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
    




    def __get_activeWeapon_dummies__(self, pf):
    
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
    




    def __player_dataset_create__(self, pf, tick_number = 1):
    
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
    




    def __insert_columns_into_player_dataframes__(stat_df, players_df):
        for col in stat_df.columns:
            if col != 'player_name':
                players_df[col] = stat_df.loc[stat_df['player_name'] == players_df['name'].iloc[0]][col].iloc[0]
        return players_df

    def __get_player_overall_statistics_without_inferno__(self, players):
        # Needed columns
        needed_stats = ['player_name', 'rating_2.0', 'DPR', 'KAST', 'Impact', 'ADR', 'KPR','total_kills', 'HS%', 'total_deaths', 'KD_ratio', 'dmgPR',
        'grenade_dmgPR', 'maps_played', 'saved_by_teammatePR', 'saved_teammatesPR','opening_kill_rating', 'team_W%_after_opening',
        'opening_kill_in_W_rounds', 'rating_1.0_all_Career', 'clutches_1on1_ratio', 'clutches_won_1on1', 'clutches_won_1on2', 'clutches_won_1on3', 'clutches_won_1on4', 'clutches_won_1on5']
        
        stats = pd.read_csv(self.PLAYER_STATS_DATA_FOLDER_PATH).drop_duplicates()
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
                players[idx] = self.__insert_columns_into_player_dataframes__(stats, players[idx])

            # If the stats dataframe does not contain the player related informations, check if the missing_players_df contains the player
            else:
                mpdf = pd.read_csv(self.MISSING_PLAYER_STATS_DATA_FOLDER_PATH)
                mpdf = mpdf[needed_stats]
                for col in mpdf.columns:
                    if col != 'player_name':
                        mpdf[col] = mpdf[col].astype('float32')
                        mpdf.rename(columns={col: "overall_" + col}, inplace=True)
                        
                # If the missing_players_df contains the player related informations, do the merge
                if len(mpdf.loc[mpdf['player_name'] == players[idx]['name'].iloc[0]]) == 1:
                    players[idx] = self.__insert_columns_into_player_dataframes__(mpdf, players[idx])

                # Else get imputed values for the player from missing_players_df and do the merge
                else:
                    first_anonim_pro_index = mpdf.index[mpdf['player_name'] == 'anonim_pro'].min()
                    mpdf.at[first_anonim_pro_index, 'player_name'] = players[idx]['name'].iloc[0]
                    mpdf.to_csv(self.MISSING_PLAYER_STATS_DATA_FOLDER_PATH, index=False)
                    players[idx] = self.__insert_columns_into_player_dataframes__(mpdf, players[idx])
            
        return players
    




    def __calculate_ct_equipment_value__(self, row):
        if row['player0_isCT']:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __calculate_t_equipment_value__(self, row):
        if row['player0_isCT'] == False:
            return row[['player0_equi_val_alive', 'player1_equi_val_alive', 'player2_equi_val_alive', 'player3_equi_val_alive', 'player4_equi_val_alive']].sum()
        else:
            return row[['player5_equi_val_alive', 'player6_equi_val_alive', 'player7_equi_val_alive', 'player8_equi_val_alive', 'player9_equi_val_alive']].sum()

    def __create_game_snapshot_dataset__(self, players, rounds, match_id):
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
        graph_data['CT_aliveNum'] = graph_data[['player0_isAlive','player1_isAlive','player2_isAlive','player3_isAlive','player4_isAlive']].sum(axis=1)
        graph_data['T_aliveNum'] = graph_data[['player5_isAlive','player6_isAlive','player7_isAlive','player8_isAlive','player9_isAlive']].sum(axis=1)

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
        graph_data['CT_equipmentValue'] = graph_data.apply(self.__calculate_ct_equipment_value__, axis=1)
        graph_data['T_equipmentValue'] = graph_data.apply(self.__calculate_t_equipment_value__, axis=1)

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





    def __add_bomb_related_information_to_game_snapshot_dataset__(self, gsndf, bombdf):
        gsndf['is_bomb_being_planted'] = 0
        gsndf['is_bomb_planted'] = 0
        gsndf['is_bomb_being_defused'] = 0
        gsndf['is_bomb_defused'] = 0
        gsndf['is_bomb_planted_at_A_site'] = 0
        gsndf['bomb_X'] = 0.0
        gsndf['bomb_Y'] = 0.0
        gsndf['bomb_Z'] = 0.0

        for index, row in bombdf.iterrows():
            if (row['bombAction'] == 'plant_begin'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_planted'] = 1

            if (row['bombAction'] == 'plant_abort'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_planted'] = 0

            if (row['bombAction'] == 'plant'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_planted'] = 0
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_planted'] = 1
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_planted_at_A_site'] = 1 if row['bombSite'] == 'A' else 0
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'bomb_X'] = row['playerX']
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'bomb_Y'] = row['playerY']
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'bomb_Z'] = row['playerZ']

            if (row['bombAction'] == 'defuse_start'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_defused'] = 1

            if (row['bombAction'] == 'defuse_aborted'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_defused'] = 0

            if (row['bombAction'] == 'defuse'):
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_being_defused'] = 0
                gsndf.loc[(gsndf['roundNum'] == row['roundNum']) & (gsndf['tick'] >= row['tick']), 'is_bomb_defused'] = 1

        return gsndf