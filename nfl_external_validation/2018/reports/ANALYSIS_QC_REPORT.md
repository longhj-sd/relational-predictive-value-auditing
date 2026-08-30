# Analysis QC Report

- no_heldout_game_in_training: True
- no_heldout_play_in_training: True
- ball_land_only_D: True
- endpoint_errors_nonnegative: True
- gain_identity: True
- coordinates_strict_field_bounds: False
- coordinates_plausible_tracking_buffer: True

Strict field-bound exceedance rows: 805 of 86,816. These are retained as plausible tracking values just outside the playing-field rectangle during pass outcomes; no row exceeds the QC buffer of x [-5, 125] and y [-5, 58.33].

|      | true_x | input_start_x | input_end_x | ball_land_x | true_y | input_start_y | input_end_y | ball_land_y |
|:-----|-------:|--------------:|------------:|------------:|-------:|--------------:|------------:|------------:|
| min  |   1.17 |          5.35 |        2.77 |        5.30 | -3.267 |         0.890 |      -0.057 |       -4.18 |
| max  | 122.78 |        116.69 |      119.96 |      122.51 | 54.870 |        51.213 |      53.140 |       57.19 |
