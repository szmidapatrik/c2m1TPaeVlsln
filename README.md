


<span style="color: cornflowerblue; font-size: 16px">MIT Sloan: <span><span style="color: white">The MIT Sloan related information are stored in the [MIT-SLOAN-README.md](MIT-SLOAN-README.md) file.<span> 

---

<span style="color: red; font-size: 16px">IMPORTANT!<span><span style="color: white"> If any errors would occure during the use of this project, read the [BUGFIX.md](BUGFIX.md) file, which contains solutions to some of the previously found problems.<span> 




### 0. ***TODO***s

The list of tasks that are currently under development or are future upgrade plans.

<div style="min-height: 40px"></div>

----


### 1. Python kernels (ASUS ROG)


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
    <td>T</td>
  </tr>
  <tr>
    <td>3</td>
    <td>CT</td>
  </tr>
</table>

<div style="min-height: 40px"></div>

----

### 4. Parser classes



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