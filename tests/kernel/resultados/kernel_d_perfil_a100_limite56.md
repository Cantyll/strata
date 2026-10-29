# STRATA, perfil do kernel D (MMVQ, uma coluna, FFN 4096 → 14336)

GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 580.82.07. STRATA `9756817`.

| tipo | só o up (µs) | gate+up (µs) | razão | gate+up sem fusão (µs) | ganho da fusão |
|---|---:|---:|---:|---:|---:|
| q5_K | 48.64 | 95.22 | 1.958 | 99.51 | +4.5% |
| q6_K | 58.47 | 128.75 | 2.202 | 119.13 | -7.5% |
| p1 | 49.91 | 93.64 | 1.876 | 101.96 | +8.9% |
| d1 | 62.28 | 113.71 | 1.826 | 126.67 | +11.4% |
| d1_linha | 51.87 | 95.64 | 1.844 | 105.75 | +10.6% |
| p2 | 54.64 | 97.51 | 1.785 | 111.32 | +14.2% |
| d2 | 74.90 | 127.45 | 1.702 | 151.76 | +19.1% |
| d2_linha | 57.64 | 102.15 | 1.772 | 117.43 | +15.0% |
| q6p1 | 83.22 | 181.81 | 2.185 | 169.27 | -6.9% |
| q6d1 | 96.48 | 210.22 | 2.179 | 169.55 | -19.3% |
| q6d1_linha | 96.36 | 176.12 | 1.828 | 169.43 | -3.8% |
| q6p2 | 57.45 | 111.89 | 1.948 | 117.09 | +4.6% |
| q6d2 | 66.12 | 123.37 | 1.866 | 134.41 | +8.9% |
| q6d2_linha | 60.30 | 112.79 | 1.870 | 122.75 | +8.8% |

## ncu: q5_K

|                                                     |   ('13', False) |    ('13', True) |
|:----------------------------------------------------|----------------:|----------------:|
| gpu__time_duration.sum                              | 48288           | 94832           |
| launch__registers_per_thread                        |    44           |    62           |
| launch__block_size                                  |   128           |   128           |
| launch__grid_size                                   | 14336           | 14336           |
| launch__occupancy_limit_registers                   |    10           |     8           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    53.565       |    43.36        |
| smsp__inst_executed.sum                             |     1.52105e+07 |     2.59482e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    71.77        |    60.27        |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    57.005       |    56.41        |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    67.355       |    58.13        |

## ncu: p1

|                                                     |   ('43', False) |    ('43', True) |
|:----------------------------------------------------|----------------:|----------------:|
| gpu__time_duration.sum                              | 49440           | 93280           |
| launch__registers_per_thread                        |    46           |    56           |
| launch__block_size                                  |   128           |   128           |
| launch__grid_size                                   | 14336           | 14336           |
| launch__occupancy_limit_registers                   |    10           |     9           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    53.59        |    48.98        |
| smsp__inst_executed.sum                             |     1.60707e+07 |     2.84713e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    74.03        |    67.425       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    55.66        |    57.36        |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    69.44        |    64.875       |

## ncu: d1

|                                                     |   ('47', False) |     ('47', True) |
|:----------------------------------------------------|----------------:|-----------------:|
| gpu__time_duration.sum                              | 60864           | 111968           |
| launch__registers_per_thread                        |    44           |     56           |
| launch__block_size                                  |   128           |    128           |
| launch__grid_size                                   | 14336           |  14336           |
| launch__occupancy_limit_registers                   |    10           |      9           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    54.83        |     49.64        |
| smsp__inst_executed.sum                             |     2.14753e+07 |      3.85352e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    79.26        |     75.515       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    45.215       |     47.775       |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    75.17        |     73.04        |

## ncu: d1_linha

|                                                     |   ('57', False) |    ('57', True) |
|:----------------------------------------------------|----------------:|----------------:|
| gpu__time_duration.sum                              | 51264           | 95200           |
| launch__registers_per_thread                        |    43           |    56           |
| launch__block_size                                  |   128           |   128           |
| launch__grid_size                                   | 14336           | 14336           |
| launch__occupancy_limit_registers                   |    10           |     9           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    54.01        |    49.08        |
| smsp__inst_executed.sum                             |     1.69308e+07 |     2.97042e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    74.91        |    68.755       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    53.695       |    56.2         |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    70.545       |    66.33        |
