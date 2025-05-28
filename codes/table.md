## Difference in performance of the Label & Track method (mean over all labellers) vs. AggPose, bolded differences are statistically significant (p < 0.05), p-values are given in the following table

| Metric | Input Features | Early GM Group | Late GM Group |
|-|-|-|-|
|||RF LSTM CNN|RF LSTM CNN|
| | Coord. |  **0.1005** **0.1226** **0.2578** | 0.0463 -0.0889 0.0071 |
| AUROC | Angles |  **0.2641** 0.0075 **0.2546** | **0.1511** **-0.1146** **-0.1625** |
| | Both |  **0.2122** **0.1091** **0.2383** | 0.0735 **-0.1737** 0.0342 |
|-|-|-|-|
| | Coord. |  **0.0890** **0.0954** **0.1865** | -0.0452 **-0.1412** 0.0008 |
| AUPRC | Angles |  **0.1623** -0.0290 **0.1500** | 0.0878 **-0.1529** **-0.2629** |
| | Both |  **0.1587** **0.0857** **0.1591** | -0.0005 **-0.2397** -0.0207 |
|-|-|-|-|
| | Coord. |  0.0671 0.0540 **0.1980** | **0.0939** -0.0122 -0.0277 |
| Accuracy | Angles |  **0.1912** -0.0235 **0.2000** | **0.0504** **-0.0693** **-0.0876** |
| | Both |  **0.1271** **0.0987** **0.1863** | 0.0426 0.0143 0.0098 |

## P values for relative performance of the Label \& Track method vs. AggPose, bolded differences are statistically significant (p < 0.05)


| Metric | Input Features | Early GM Group | Late GM Group |
|-|-|-|-|
|||RF LSTM CNN|RF LSTM CNN|
| | Coord. | **0.0213**  **0.0016**  **0.0000**  | 0.3859  0.0908  0.8785 |
| AUROC | Angles | **0.0000**  0.8431  **0.0000**  | **0.0027**  **0.0192**  **0.0030** |
| | Both | **0.0000** | **0.0069** | **0.0000** | | 0.1381 | **0.0004** | 0.5183 |
|-|-|-|-|
| | Coord. | **0.0070** | **0.0036** | **0.0000** | | 0.2423 | **0.0063** | 0.9875 |
| AUPRC | Angles | **0.0000**  0.3985  **0.0000**  | 0.0909  **0.0037**  **0.0000** |
| | Both | **0.0000**  **0.0108**  **0.0000**  | 0.9901  **0.0000**  0.7011 |
|-|-|-|-|
| | Coord. | 0.0547  0.0840  **0.0000**  | **0.0031**  0.6917  0.4870 |
| Accuracy | Angles | **0.0000**  0.4485  **0.0000**  | **0.0427**  **0.0248**  **0.0043** |
| | Both | **0.0005**  **0.0072**  **0.0000**  | 0.1324  0.6787  0.7724 |