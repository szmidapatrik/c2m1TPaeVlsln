# Counter Strike Player Action Evaluation Using Graph Neual Networks

### ***TODO***s

The list of tasks that are currently under development or are future upgrade plans.

  - Tabular snapshot creator: improve time by iterating *round-by-round* on the ***damages*** dataframe instead of *row-by-row*

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