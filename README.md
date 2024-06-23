# Counter Strike Player Action Evaluation Using Graph Neual Networks

### Python kernels
  - 3.11: *awpy* is installed on it for CSGO match parsing.
  - 3.12: *awpy2* is installed on it for CS2 match parsing.

---

### *awpy* and *awpy2* parsed tables compare

| *awpy*        | *awpy2*       |
| ------------- | ------------- |
| bombEvents    | bomb          |
| damages       | damages       |
| -             | events        |
| flashes       | -             |
| frames        | -             |
| grenades      | grenades      |
| -             | header        |
| -             | infernos      |
| kills         | kills         |
| -             | parse-ticks   |
| -             | parser        |
| -             | path          |
| playerFrames  | -             |
| rounds        | rounds        |
| -             | smokes        |
| -             | ticks         |
| -             | verbose       |
| weaponFires   | weapon_fires  |