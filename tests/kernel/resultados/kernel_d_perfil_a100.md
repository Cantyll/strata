# STRATA, perfil do kernel D (MMVQ, uma coluna, FFN 4096 → 14336)

GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 580.82.07. STRATA `0270529`.

| tipo | só o up (µs) | gate+up (µs) | razão | gate+up sem fusão (µs) | ganho da fusão |
|---|---:|---:|---:|---:|---:|
| q5_K | 48.42 | 95.02 | 1.962 | 99.06 | +4.3% |
| q6_K | 58.25 | 128.55 | 2.207 | 118.93 | -7.5% |
| p1 | 49.67 | 93.65 | 1.885 | 101.53 | +8.4% |
| d1 | 62.27 | 112.67 | 1.809 | 126.17 | +12.0% |
| d1_linha | 51.56 | 100.98 | 1.958 | 105.34 | +4.3% |
| p2 | 54.33 | 97.29 | 1.791 | 110.85 | +13.9% |
| d2 | 74.62 | 133.31 | 1.787 | 151.19 | +13.4% |
| d2_linha | 57.39 | 101.94 | 1.776 | 116.98 | +14.8% |
| q6p1 | 82.65 | 181.58 | 2.197 | 169.01 | -6.9% |
| q6d1 | 96.01 | 212.87 | 2.217 | 168.74 | -20.7% |
| q6d1_linha | 83.73 | 173.03 | 2.067 | 168.78 | -2.5% |
| q6p2 | 57.42 | 111.72 | 1.946 | 116.76 | +4.5% |
| q6d2 | 65.72 | 122.78 | 1.868 | 132.24 | +7.7% |
| q6d2_linha | 60.07 | 112.54 | 1.873 | 122.54 | +8.9% |

## ncu: q5_K

|                                                     |   ('13', False) |    ('13', True) |
|:----------------------------------------------------|----------------:|----------------:|
| gpu__time_duration.sum                              | 48096           | 94624           |
| launch__registers_per_thread                        |    44           |    62           |
| launch__block_size                                  |   128           |   128           |
| launch__grid_size                                   | 14336           | 14336           |
| launch__occupancy_limit_registers                   |    10           |     8           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    53.56        |    43.355       |
| smsp__inst_executed.sum                             |     1.52105e+07 |     2.59482e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    71.55        |    60.125       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    57.22        |    56.54        |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    67.175       |    58.015       |

## ncu: p1

|                                                     |   ('43', False) |    ('43', True) |
|:----------------------------------------------------|----------------:|----------------:|
| gpu__time_duration.sum                              | 49120           | 92832           |
| launch__registers_per_thread                        |    46           |    56           |
| launch__block_size                                  |   128           |   128           |
| launch__grid_size                                   | 14336           | 14336           |
| launch__occupancy_limit_registers                   |    10           |     9           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    53.58        |    49.01        |
| smsp__inst_executed.sum                             |     1.60707e+07 |     2.84713e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    74           |    67.355       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    56           |    57.62        |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    69.53        |    64.915       |

## ncu: d1

|                                                     |   ('47', False) |     ('47', True) |
|:----------------------------------------------------|----------------:|-----------------:|
| gpu__time_duration.sum                              | 60640           | 110864           |
| launch__registers_per_thread                        |    44           |     56           |
| launch__block_size                                  |   128           |    128           |
| launch__grid_size                                   | 14336           |  14336           |
| launch__occupancy_limit_registers                   |    10           |      9           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    54.83        |     49.55        |
| smsp__inst_executed.sum                             |     2.14753e+07 |      3.86212e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    79.25        |     76.145       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    45.375       |     48.245       |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    75.145       |     73.74        |

## ncu: d1_linha

|                                                     |   ('57', False) |     ('57', True) |
|:----------------------------------------------------|----------------:|-----------------:|
| gpu__time_duration.sum                              | 51024           | 100336           |
| launch__registers_per_thread                        |    43           |     61           |
| launch__block_size                                  |   128           |    128           |
| launch__grid_size                                   | 14336           |  14336           |
| launch__occupancy_limit_registers                   |    10           |      8           |
| sm__warps_active.avg.pct_of_peak_sustained_active   |    54.035       |     43.41        |
| smsp__inst_executed.sum                             |     1.69308e+07 |      2.97185e+07 |
| smsp__issue_active.avg.pct_of_peak_sustained_active |    74.805       |     64.795       |
| dram__throughput.avg.pct_of_peak_sustained_elapsed  |    53.93        |     53.32        |
| sm__throughput.avg.pct_of_peak_sustained_elapsed    |    70.4         |     62.57        |
