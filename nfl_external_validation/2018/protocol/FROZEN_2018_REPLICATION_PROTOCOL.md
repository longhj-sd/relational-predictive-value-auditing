# Frozen 2018 Replication Protocol

Freeze timestamp: 2026-08-30T12:45:32

This protocol was written and hashed before fitting the final 2018 HGB models or viewing primary 2018 RPVA results.

## Locked Configuration

```yaml
data_source: official NFL Big Data Bowl 2021 competition files
target_receiver_source: tombliss/nfl-big-data-bowl-2021-bonus targetedReceiver.csv
target_label_mode: OFFICIAL_BONUS primary; reconstructed only for QC/sensitivity if
  needed
event_definitions:
  snap: earliest ball_snap
  forward: earliest pass_forward after snap
  arrival: earliest pass_arrived after forward
temporal_split:
  train: weeks 1-11
  validation: weeks 12-13
  heldout: weeks 14-17
coordinate_normalization: prespecified left-to-right coordinate normalization
primary_defender_set:
- CB
- DB
- FS
- S
- SS
expanded_defender_set:
- CB
- DB
- FS
- ILB
- LB
- MLB
- OLB
- S
- SS
roles:
  R: official targeted receiver
  N: nearest coverage-eligible defender to R at pass_forward
  O: remaining coverage-eligible defenders, within-play mean
features:
  A:
  - n_input_frames
  - input_start_x
  - input_start_y
  - input_end_x
  - input_end_y
  - input_delta_x
  - input_delta_y
  - input_path_length
  - input_mean_speed
  - input_max_speed
  - input_mean_acceleration
  - input_max_acceleration
  - input_mean_dir
  - input_mean_o
  - play_direction
  - player_side
  - player_role
  - player_position
  H_add:
  - num_frames_output
  D_add:
  - ball_land_x
  - ball_land_y
  - distance_to_ball_land_start
  - distance_to_ball_land_end
  - change_in_distance_to_ball_land
loss: Euclidean endpoint error in yards
HGB: sklearn HistGradientBoostingRegressor defaults, random_state 20260730; deterministic
  preprocessing
ExtraTrees: skipped unless exact submitted final configuration is recoverable
bootstrap:
  method: percentile game-cluster bootstrap
  replicates: 2000
  seed: 20260730
pseudo_nearest:
  permutations: 5000
  seed: 20260731
  p_value: add-one corrected two-sided absolute statistic
estimands:
- G_R
- G_N
- G_O
- lambda_D
- delta_RN
- delta_RO
sensitivity_analyses:
- primary defender set
- expanded defender set
- +/-1 football endpoint
- exclude ambiguous reconstructed-target QC flags

```

## Raw Data Hashes

| source                                      | file                       | sha256                                                           |      size |   row_count |
|:--------------------------------------------|:---------------------------|:-----------------------------------------------------------------|----------:|------------:|
| official NFL Big Data Bowl 2021 competition | games.csv                  | 17d4038615861cc115817c7d295416f34206098a1f54ac757db405ba8eb99a29 |     10451 |         253 |
| official NFL Big Data Bowl 2021 competition | nfl-big-data-bowl-2021.zip | cf86da5b3415eabde5f3f1b737618fe4850b93557f4daa7d40612fea8d570f7c | 421153222 |             |
| official NFL Big Data Bowl 2021 competition | players.csv                | 8d91099df389615ca31901ca2130cff0073969bfca11581d05e558eb89b6d446 |     71358 |        1303 |
| official NFL Big Data Bowl 2021 competition | plays.csv                  | c837c1a1e7874fffc15d265a849f43d12ba37fc68fc8a806dd560fa3493ef960 |   4979613 |       19239 |
| official NFL Big Data Bowl 2021 competition | week1.csv                  | 143897386975fd66d7010a9a84b74155e2e384b67a767a41fa3ca620a8fca11e | 125378352 |      986022 |
| official NFL Big Data Bowl 2021 competition | week10.csv                 | 070b4b894c09c407aa5586f4f1f873643de74ed62126c5091bea9311b82674dc | 122774466 |      964889 |
| official NFL Big Data Bowl 2021 competition | week11.csv                 | 1232bb436622fd2e161be76f79834d9e7e96796092bfd3879c16424143b4a2b1 | 118578223 |      932240 |
| official NFL Big Data Bowl 2021 competition | week12.csv                 | 21a4ad802d424177b516fbf6a0ed90548da0c0df4b1b894577d71b93381e9869 | 130340962 |     1024868 |
| official NFL Big Data Bowl 2021 competition | week13.csv                 | 1b56a3c60f482b401df30da081477b932fb06c39db0650aca33088eda465aa58 | 149141557 |     1172517 |
| official NFL Big Data Bowl 2021 competition | week14.csv                 | c5969a10c65598cda3fd1a740e49f888487d5d519fa869583f37a93b1fdad574 | 147823618 |     1161644 |
| official NFL Big Data Bowl 2021 competition | week15.csv                 | 91441c95f8ffe27d4e02475940b99e2c62bee6243ec4f1de5b228beca8738bd0 | 137613898 |     1081222 |
| official NFL Big Data Bowl 2021 competition | week16.csv                 | d820a86eb58366d7dc00f4d9312eafe5be9c8dba0cd820857d597d7a8da8f7e1 | 145614217 |     1144037 |
| official NFL Big Data Bowl 2021 competition | week17.csv                 | 87eecbcd22bcd8988ffaf462a808e920426df25b6ea4e810319c315b2c069b3b | 133455817 |     1049265 |
| official NFL Big Data Bowl 2021 competition | week2.csv                  | 7d37848533ef092f32c3e46fe4b7f8a642003ef8e4cffd15e0ccd26700528927 | 156777982 |     1231793 |
| official NFL Big Data Bowl 2021 competition | week3.csv                  | ce0a1a08a96f7ad6653f244ecd7fba82498e23080a962b4ac79b05ddc97aecc4 | 148630299 |     1168345 |
| official NFL Big Data Bowl 2021 competition | week4.csv                  | 4cda6f9180c90d5cde8f240fed76103dc3d7d04db31889300fb506c8d4177549 | 153463956 |     1205527 |
| official NFL Big Data Bowl 2021 competition | week5.csv                  | 2752292e72866a9d5b7e685a804b4a5c044e3e7db857ede9c6c9f894f5a55bc8 | 149220235 |     1171908 |
| official NFL Big Data Bowl 2021 competition | week6.csv                  | e76b399554ad15c26a39be54088ab3ec0b93f146a8f4701bd161a6ba3291700e | 136472235 |     1072563 |
| official NFL Big Data Bowl 2021 competition | week7.csv                  | 6ec8e8e21680bcc3504e41cbc543b16650e62284264d8bea999ed2a54fac11fe | 124905285 |      982583 |
| official NFL Big Data Bowl 2021 competition | week8.csv                  | 45d7c9da242972a29967a929a4783caa3e08f0b6e0e770d2c81785650635a1ad | 127376389 |     1001501 |
| official NFL Big Data Bowl 2021 competition | week9.csv                  | 1a77506f52beb12fc9b72b0dc8f07a9839ee988f04464ba65707f5bc4e4dadae | 121946210 |      958464 |
| tombliss BDB2021 bonus                      | coverages_week1.csv        | 5617e9385b95198feb364cd330a9ef360561c1a09882136fcbe169e8d679b52a |     32350 |        1028 |
| tombliss BDB2021 bonus                      | targetedReceiver.csv       | f3fd9eff3870a211bf1b7111d634ff63e61cb450c6d1d3c5959caf09a3b8b8db |    471441 |       19239 |

## Interpretation Rules

The 2018 analysis is an external-season RPVA replication using harmonized BDB2021 data. Proximity-based N is a nearest observed coverage-defender proxy, not an assigned defender. Destination and endpoint variables are retrospective audit-state inputs, not deployment inputs. Weak, null, or discordant results do not trigger post-freeze redefinition.
