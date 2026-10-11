### Workers, at the original 3.0 m aisle

Success as n/50 with a Wilson 95% interval for a single policy, or as the mean over three training seeds with the range in brackets.

| workers | D* + RPP | D* + MPC | D* + TD3 | TD3 end-to-end |
|---|---|---|---|---|
| 0 | 50/50 [93, 100] | 50/50 [93, 100] | 47.7/50 (47-49) | 44.7/50 (43-47) |
| 4 | 50/50 [93, 100] | 50/50 [93, 100] | 48.0/50 (47-49) | 43.3/50 (42-45) |
| 8 | 50/50 [93, 100] | 50/50 [93, 100] | 46.3/50 (45-48) | 45.3/50 (44-46) |

| workers | D* + RPP median s | D* + MPC median s | D* + TD3 median s | TD3 end-to-end median s |
|---|---|---|---|---|
| 0 | 22.9 | 23.5 | 19.1 | 18.9 |
| 4 | 25.5 | 24.2 | 19.9 | 19.8 |
| 8 | 28.4 | 24.6 | 20.1 | 20.8 |

Failures by cause, pooled over seeds (workers):

| workers | D* + RPP | D* + MPC | D* + TD3 | TD3 end-to-end |
|---|---|---|---|---|
| 0 | none | none | 7 timeout | 16 timeout |
| 4 | none | none | 4 worker, 2 timeout | 1 rack, 9 worker, 10 timeout |
| 8 | none | none | 11 worker | 10 worker, 4 timeout |

### Density, at 4 workers

A narrower aisle adds rows rather than opening a bypass, so occupancy rises from 12.3% to 18.4% and free navigable cells fall from 68% to 52%. One policy per method here. **The TD3 policies were trained only at the 3.0 m, four-row layout, so every narrower column is out of distribution for them and in distribution for the two classical controllers.**

| aisle m | D* + RPP | D* + MPC | D* + TD3 | TD3 end-to-end |
|---|---|---|---|---|
| 3.0 (4 rows) | 50/50 [93, 100] | 50/50 [93, 100] | 48.0/50 (47-49) | 43.3/50 (42-45) |
| 2.4 (5 rows) | 49/50 [90, 100] | 50/50 [93, 100] | 47/50 [84, 98] | 41/50 [69, 90] |
| 2.0 (5 rows) | 49/50 [90, 100] | 50/50 [93, 100] | 43/50 [74, 93] | 39/50 [65, 87] |
| 1.6 (6 rows) | 49/50 [90, 100] | 50/50 [93, 100] | 34/50 [54, 79] | 22/50 [31, 58] |
| 1.3 (6 rows) | 50/50 [93, 100] | 50/50 [93, 100] | 47/50 [84, 98] | 32/50 [50, 76] |

| aisle m | D* + RPP median s | D* + MPC median s | D* + TD3 median s | TD3 end-to-end median s |
|---|---|---|---|---|
| 3.0 | 25.5 | 24.2 | 19.9 | 19.8 |
| 2.4 | 26.7 | 23.5 | 21.2 | 20.6 |
| 2.0 | 25.5 | 25.1 | 24.2 | 23.2 |
| 1.6 | 28.5 | 26.0 | 27.2 | 23.8 |
| 1.3 | 30.2 | 26.9 | 22.3 | 22.1 |

Failures by cause, pooled over seeds (aisle m):

| aisle m | D* + RPP | D* + MPC | D* + TD3 | TD3 end-to-end |
|---|---|---|---|---|
| 3.0 | none | none | 4 worker, 2 timeout | 1 rack, 9 worker, 10 timeout |
| 2.4 | 1 timeout | none | 3 worker | 9 timeout |
| 2.0 | 1 timeout | none | 1 rack, 2 worker, 4 timeout | 3 worker, 8 timeout |
| 1.6 | 1 timeout | none | 8 rack, 1 worker, 7 timeout | 1 rack, 3 worker, 24 timeout |
| 1.3 | none | none | 2 rack, 1 worker | 2 rack, 4 worker, 12 timeout |

