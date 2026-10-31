# Projection robustness: grid_v1 vs grid_v3 (train+val, 169 judgments)

| | grid_v1 | grid_v3 (Stanza) |
|---|---:|---:|
| grid sentences | 11,047 | 8,515 |
| median sentence length (chars) | 144 | 180 |
| unlabeled (no C region) | 1,577 (14.3%) | 1,308 (15.4%) |
| covered | 9,470 (85.7%) | 7,207 (84.6%) |
| AMBIGUOUS_PROJECTION | 1,911 (17.3%) | 1,349 (15.8%) |
|   co-extensive multi-role | 1,165 | 858 |
|   nested | 643 | 412 |
|   other | 103 | 79 |
| ambiguous as share of covered | 20.2% | 18.7% |
| eligible single-label | 7,559 (68.4%) | 5,858 (68.8%) |
| identity holds (train, val, pooled) | True | True |
| judgments with |Δ ambiguity rate| > 5 pts vs grid_v1 | — | 35 / 169 |

## Per split

- **train / grid_v1**: grid 9,283; unlabeled 1,451; covered 7,832; ambiguous 1,534 {'co_extensive_multi_role': 914, 'other': 79, 'nested': 541}; eligible 6,298; identity True
- **train / grid_v3**: grid 7,133; unlabeled 1,203; covered 5,930; ambiguous 1,083 {'co_extensive_multi_role': 670, 'other': 66, 'nested': 347}; eligible 4,847; identity True
- **val / grid_v1**: grid 1,764; unlabeled 126; covered 1,638; ambiguous 377 {'co_extensive_multi_role': 251, 'nested': 102, 'other': 24}; eligible 1,261; identity True
- **val / grid_v3**: grid 1,382; unlabeled 105; covered 1,277; ambiguous 266 {'co_extensive_multi_role': 188, 'nested': 65, 'other': 13}; eligible 1,011; identity True

## Moved > 5 points (35): direction v3 lower 28, higher 7; of which v3 grid < 20 sentences: 9

| case | split | v1 | v3 | Δ | v1 n | v3 n |
|---|---|---:|---:|---:|---:|---:|
| C0077 | train | 39.13 | 6.67 | -32.46 | 46 | 30 |
| C0089 | train | 46.34 | 20.0 | -26.34 | 41 | 20 |
| C0135 | train | 70.59 | 44.44 | -26.15 | 17 | 9 |
| C0143 | train | 35.42 | 60.0 | +24.58 | 48 | 10 |
| C0146 | train | 55.56 | 31.82 | -23.74 | 45 | 22 |
| C0153 | train | 22.73 | 0.0 | -22.73 | 22 | 2 |
| C0091 | train | 76.92 | 55.56 | -21.36 | 13 | 9 |
| C0065 | val | 53.33 | 33.33 | -20.0 | 30 | 21 |
| C0176 | train | 33.33 | 15.38 | -17.95 | 120 | 39 |
| C0011 | train | 15.38 | 0.0 | -15.38 | 26 | 8 |
| C0129 | train | 30.77 | 15.79 | -14.98 | 52 | 38 |
| C0151 | train | 28.07 | 42.62 | +14.55 | 57 | 61 |
| C0183 | train | 30.95 | 16.67 | -14.28 | 42 | 24 |
| C0005 | train | 44.26 | 32.69 | -11.57 | 61 | 52 |
| C0169 | train | 62.16 | 51.72 | -10.44 | 37 | 29 |
| C0004 | val | 29.7 | 20.0 | -9.7 | 101 | 90 |
| C0173 | train | 40.91 | 31.25 | -9.66 | 22 | 16 |
| C0136 | val | 27.69 | 18.18 | -9.51 | 65 | 22 |
| C0110 | train | 36.36 | 45.83 | +9.47 | 22 | 24 |
| C0185 | train | 25.56 | 16.22 | -9.34 | 90 | 74 |
| C0117 | val | 25.93 | 16.67 | -9.26 | 54 | 36 |
| C0100 | train | 100.0 | 90.91 | -9.09 | 14 | 11 |
| C0074 | train | 34.04 | 25.0 | -9.04 | 47 | 36 |
| C0158 | train | 58.82 | 50.0 | -8.82 | 17 | 14 |
| C0069 | val | 26.76 | 18.18 | -8.58 | 142 | 110 |
| C0167 | train | 12.5 | 4.0 | -8.5 | 32 | 25 |
| C0125 | train | 49.17 | 40.91 | -8.26 | 360 | 286 |
| C0118 | train | 14.63 | 22.22 | +7.59 | 41 | 27 |
| C0116 | train | 15.33 | 22.22 | +6.89 | 137 | 72 |
| C0142 | train | 18.67 | 12.12 | -6.55 | 75 | 66 |
| C0127 | train | 15.38 | 21.43 | +6.05 | 39 | 28 |
| C0172 | train | 11.7 | 5.88 | -5.82 | 94 | 68 |
| C0111 | train | 47.22 | 41.94 | -5.28 | 36 | 31 |
| C0061 | train | 9.09 | 14.29 | +5.2 | 11 | 7 |
| C0192 | train | 9.09 | 4.08 | -5.01 | 55 | 49 |
