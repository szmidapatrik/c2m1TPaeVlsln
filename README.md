# Counter Strike Player Action Evaluation Using Graph Neual Networks


<span style="color: cornflowerblue; font-size: 16px">MIT Sloan: <span><span style="color: white">The MIT Sloan related information are stored in the [MIT-SLOAN-README.md](MIT-SLOAN-README.md) file.<span> 

---

<span style="color: red; font-size: 16px">IMPORTANT!<span><span style="color: white"> If any errors would occure during the use of this project, read the [BUGFIX.md](BUGFIX.md) file, which contains solutions to some of the previously found problems.<span> 




### 0. ***TODO***s

The list of tasks that are currently under development or are future upgrade plans.

<div style="min-height: 40px"></div>

----


### 1. Python kernels (ASUS ROG Zepgyrus G16)
  - 3.11: *awpy* is installed on it for CSGO match parsing.
  - 3.12: *awpy2a9* is installed on it for CS2 match parsing.

<div style="min-height: 40px"></div>

----

### 2. Parsed match tables comparison

<table>
  <tr>
    <th style="min-width: 200px">awpy</th>
    <th style="min-width: 200px">awpy2</th>
  </tr>
  <tr>
    <td>bombEvents</td>
    <td>bomb</td>
  </tr>
  <tr>
    <td>damages</td>
    <td>damages</td>
  </tr>
  <tr>
    <td>-</td>
    <td>events</td>
  </tr>
  <tr>
    <td>flashes</td>
    <td>-</td>
  </tr>
  <tr>
    <td>frames</td>
    <td>-</td>
  </tr>
  <tr>
    <td>grenades</td>
    <td>grenades</td>
  </tr>
  <tr>
    <td>-</td>
    <td>header</td>
  </tr>
  <tr>
    <td>-</td>
    <td>infernos</td>
  </tr>
  <tr>
    <td>kills</td>
    <td>kills</td>
  </tr>
  <tr>
    <td>playerFrames</td>
    <td>ticks</td>
  </tr>
  <tr>
    <td>rounds</td>
    <td>rounds</td>
  </tr>
  <tr>
    <td>-</td>
    <td>smokes</td>
  </tr>
  <tr>
    <td>weaponFires</td>
    <td>weapon_fires</td>
  </tr>
</table>

<div style="min-height: 40px"></div>

----

### 3. Round win informations

Winning team by number:

<table>
  <tr>
    <th style="min-width: 200px">Winning team number</th>
    <th style="min-width: 200px">Meaning</th>
  </tr>
  <tr>
    <td>2</td>
    <td>T - Terrorists</td>
  </tr>
  <tr>
    <td>3</td>
    <td>CT - Counter-Terrorists</td>
  </tr>
</table>

<div style="min-height: 40px"></div>

----

### 4. CS2 parser classes

Classes:

  - CS2_tabular_snapshots: responsible for creating tabular snapshots from a demo file.
  - CS2_dictionary: responsible for creating the normalizing dictionary for the scaling.
  - CS2_map: responsible for normalizing map node coordinates and saving the scaler for later use.

Function name conventions:

  - <div>__PREP__: Preparations are done in these functions, unrelated to the actual functionality/purpose.</div>
  - <div>__EXT__: External functions, samller code parts are organized here for better readability.</div>


  - <div>_INIT_: Initial functions, initial steps are done in these parts.</div>
  - <div>_PLAYER_: These functions work on databases that contain information about the players.</div>
  - <div>_TABULAR_: These functions work on databases that contain information about the game snapshots.</div>
  - <div>_FINAL_: Finalizing the function.</div>

<div style="min-height: 40px"></div>

----

### 5. Token

Token value orders plan:

  - 0: *(3 digits)* Token version 
  - 1: *(**n** digits)* CT position encodings 
  - 2: *(**n** digits)* T position encodings 
  - 3: *(1 digit)* CT buy type (0-3) 
  - 4: *(3 digit)* T buy type (0-3) 
  - 5: *(2 digits)* CT score (with length of 2) 
  - 6: *(2 digits)* T score (with length of 2) 
  - 7: *(1 digit)* CT wins the round (1 - true, 0 - false) 

---

### HeteroData player features column order

'CT0_X',
 'CT0_Y',
 'CT0_Z',
 'CT0_pitch',
 'CT0_yaw',
 'CT0_velocity_X',
 'CT0_velocity_Y',
 'CT0_velocity_Z',
 'CT0_health',
 'CT0_armor_value',
 'CT0_active_weapon_magazine_size',
 'CT0_active_weapon_ammo',
 'CT0_active_weapon_magazine_ammo_left_%',
 'CT0_active_weapon_max_ammo',
 'CT0_total_ammo_left',
 'CT0_active_weapon_total_ammo_left_%',
 'CT0_flash_duration',
 'CT0_flash_max_alpha',
 'CT0_balance',
 'CT0_current_equip_value',
 'CT0_round_start_equip_value',
 'CT0_cash_spent_this_round',
 'CT0_is_alive',
 'CT0_is_CT',
 'CT0_is_shooting',
 'CT0_is_crouching',
 'CT0_is_ducking',
 'CT0_is_duck_jumping',
 'CT0_is_walking',
 'CT0_is_spotted',
 'CT0_is_scoped',
 'CT0_is_defusing',
 'CT0_is_reloading',
 'CT0_is_in_bombsite',
 'CT0_zoom_lvl',
 'CT0_velo_modifier',
 'CT0_stat_kills',
 'CT0_stat_HS_kills',
 'CT0_stat_opening_kills',
 'CT0_stat_MVPs',
 'CT0_stat_deaths',
 'CT0_stat_opening_deaths',
 'CT0_stat_assists',
 'CT0_stat_flash_assists',
 'CT0_stat_damage',
 'CT0_stat_weapon_damage',
 'CT0_stat_nade_damage',
 'CT0_stat_survives',
 'CT0_stat_KPR',
 'CT0_stat_ADR',
 'CT0_stat_DPR',
 'CT0_stat_HS%',
 'CT0_stat_SPR',
 'CT0_inventory_C4',
 'CT0_inventory_Taser',
 'CT0_inventory_USP-S',
 'CT0_inventory_P2000',
 'CT0_inventory_Glock-18',
 'CT0_inventory_Dual Berettas',
 'CT0_inventory_P250',
 'CT0_inventory_Tec-9',
 'CT0_inventory_CZ75 Auto',
 'CT0_inventory_Five-SeveN',
 'CT0_inventory_Desert Eagle',
 'CT0_inventory_R8 Revolver',
 'CT0_inventory_MAC-10',
 'CT0_inventory_MP9',
 'CT0_inventory_MP7',
 'CT0_inventory_MP5-SD',
 'CT0_inventory_UMP-45',
 'CT0_inventory_PP-Bizon',
 'CT0_inventory_P90',
 'CT0_inventory_Nova',
 'CT0_inventory_XM1014',
 'CT0_inventory_Sawed-Off',
 'CT0_inventory_MAG-7',
 'CT0_inventory_M249',
 'CT0_inventory_Negev',
 'CT0_inventory_FAMAS',
 'CT0_inventory_Galil AR',
 'CT0_inventory_AK-47',
 'CT0_inventory_M4A4',
 'CT0_inventory_M4A1-S',
 'CT0_inventory_SG 553',
 'CT0_inventory_AUG',
 'CT0_inventory_SSG 08',
 'CT0_inventory_AWP',
 'CT0_inventory_G3SG1',
 'CT0_inventory_SCAR-20',
 'CT0_inventory_HE Grenade',
 'CT0_inventory_Flashbang',
 'CT0_inventory_Smoke Grenade',
 'CT0_inventory_Incendiary Grenade',
 'CT0_inventory_Molotov',
 'CT0_inventory_Decoy Grenade',
 'CT0_active_weapon_C4',
 'CT0_active_weapon_Knife',
 'CT0_active_weapon_Taser',
 'CT0_active_weapon_USP-S',
 'CT0_active_weapon_P2000',
 'CT0_active_weapon_Glock-18',
 'CT0_active_weapon_Dual Berettas',
 'CT0_active_weapon_P250',
 'CT0_active_weapon_Tec-9',
 'CT0_active_weapon_CZ75 Auto',
 'CT0_active_weapon_Five-SeveN',
 'CT0_active_weapon_Desert Eagle',
 'CT0_active_weapon_R8 Revolver',
 'CT0_active_weapon_MAC-10',
 'CT0_active_weapon_MP9',
 'CT0_active_weapon_MP7',
 'CT0_active_weapon_MP5-SD',
 'CT0_active_weapon_UMP-45',
 'CT0_active_weapon_PP-Bizon',
 'CT0_active_weapon_P90',
 'CT0_active_weapon_Nova',
 'CT0_active_weapon_XM1014',
 'CT0_active_weapon_Sawed-Off',
 'CT0_active_weapon_MAG-7',
 'CT0_active_weapon_M249',
 'CT0_active_weapon_Negev',
 'CT0_active_weapon_FAMAS',
 'CT0_active_weapon_Galil AR',
 'CT0_active_weapon_AK-47',
 'CT0_active_weapon_M4A4',
 'CT0_active_weapon_M4A1-S',
 'CT0_active_weapon_SG 553',
 'CT0_active_weapon_AUG',
 'CT0_active_weapon_SSG 08',
 'CT0_active_weapon_AWP',
 'CT0_active_weapon_G3SG1',
 'CT0_active_weapon_SCAR-20',
 'CT0_active_weapon_HE Grenade',
 'CT0_active_weapon_Flashbang',
 'CT0_active_weapon_Smoke Grenade',
 'CT0_active_weapon_Incendiary Grenade',
 'CT0_active_weapon_Molotov',
 'CT0_active_weapon_Decoy Grenade',
 'CT0_hltv_rating_2.0',
 'CT0_hltv_DPR',
 'CT0_hltv_KAST',
 'CT0_hltv_Impact',
 'CT0_hltv_ADR',
 'CT0_hltv_KPR',
 'CT0_hltv_total_kills',
 'CT0_hltv_HS%',
 'CT0_hltv_total_deaths',
 'CT0_hltv_KD_ratio',
 'CT0_hltv_dmgPR',
 'CT0_hltv_grenade_dmgPR',
 'CT0_hltv_maps_played',
 'CT0_hltv_saved_by_teammatePR',
 'CT0_hltv_saved_teammatesPR',
 'CT0_hltv_opening_kill_rating',
 'CT0_hltv_team_W%_after_opening',
 'CT0_hltv_opening_kill_in_W_rounds',
 'CT0_hltv_rating_1.0_all_Career',
 'CT0_hltv_clutches_1on1_ratio',
 'CT0_hltv_clutches_won_1on1',
 'CT0_hltv_clutches_won_1on2',
 'CT0_hltv_clutches_won_1on3',
 'CT0_hltv_clutches_won_1on4',
 'CT0_hltv_clutches_won_1on5'