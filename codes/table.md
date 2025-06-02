## Difference in performance of the Label & Track method (mean over all labellers) vs. AggPose, bolded differences are statistically significant (p < 0.05), p-values are given in the following table. Values marked with an asterisk were calculated using a t-test, all other values used the Mann-Whitney U test.

| Metric | Input Features | Early GM Group | Late GM Group |
|-|-|-|-|
|||RF LSTM CNN|RF LSTM CNN|
| | Coord. |  **0.1005*** **0.1226*** **0.2578** | 0.0463 -0.0889* 0.0071 |
| | Angles |  **0.2641** 0.0075* **0.2546** | **0.1511** **-0.1146*** **-0.1625** |
| | Both |  **0.2122** **0.1091*** **0.2383** | 0.0735* **-0.1737*** 0.0342* |
|-|-|-|-|
| | Coord. |  **0.0890*** **0.0954** **0.1865** | -0.0452 **-0.1412** 0.0008 |
| | Angles |  **0.1623** -0.0290 **0.1500** | 0.0878 **-0.1529** **-0.2629** |
| | Both |  **0.1587** **0.0857** **0.1591** | -0.0005 **-0.2397** -0.0207 |
|-|-|-|-|
| | Coord. |  0.0671* **0.0540** **0.1980** | **0.0939** -0.0122* -0.0277 |
| | Angles |  **0.1912*** -0.0235* **0.2000*** | **0.0504** **-0.0693*** **-0.0876** |
| | Both |  **0.1271** **0.0987*** **0.1863** | 0.0426 0.0143* 0.0098 |
## P values for relative performance of the Label \& Track method vs. AggPose, bolded differences are statistically significant (p < 0.05). Values marked with an asterisk were calculated using a t-test, all other values used the Mann-Whitney U test.


| Metric | Input Features | Early GM Group | Late GM Group |
|-|-|-|-|
|||RF LSTM CNN|RF LSTM CNN|
| | Coord. |  **0.0213*** **0.0016*** **0.0000** | 0.3728 0.0908* 0.9536 |
| | Angles |  **0.0000** 0.8431* **0.0000** | **0.0075** **0.0192*** **0.0051** |
| | Both |  **0.0000** **0.0069*** **0.0000** | 0.1381* **0.0004*** 0.5183* |
|-|-|-|-|
| | Coord. |  **0.0070*** **0.0040** **0.0000** | 0.3200 **0.0030** 0.8771 |
| | Angles |  **0.0000** 0.4397 **0.0000** | 0.1813 **0.0040** **0.0000** |
| | Both |  **0.0000** **0.0107** **0.0000** | 1.0000 **0.0000** 0.7548 |
|-|-|-|-|
| | Coord. |  0.0547* **0.0405** **0.0000** | **0.0048** 0.6917* 0.8148 |
| | Angles |  **0.0000*** 0.4485* **0.0000*** | **0.0325** **0.0248*** **0.0071** |
| | Both |  **0.0008** **0.0072*** **0.0000** | 0.0742 0.6787* 0.6325 |