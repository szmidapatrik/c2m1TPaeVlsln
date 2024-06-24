# Counter Strike Player Action Evaluation Using Graph Neual Networks

### 1. Python kernels (ASUS ROG Zepgyrus G16)
  - 3.11: *awpy* is installed on it for CSGO match parsing.
  - 3.12: *awpy2* is installed on it for CS2 match parsing.

---

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

---

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
