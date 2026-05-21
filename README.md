# Fraud_Detection

A Python-based data pipeline designed to detect fraudulent transactions using a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset).

## 📌 Project Overview
- **Objective:** Evaluate and analyze a custom fraud detection logic against real-world transaction data.
- **Methodology:** 1. Calculate risk scores for each transaction based on custom-defined fraud detection features.
  2. Aggregate these scores into a column named `risk_score_new`.
  3. Apply a threshold of `0.6`: if the final score exceeds this threshold, the transaction is flagged as fraud (`fraud_label_new = 1`).
- **Evaluation:** Compare the pipeline's predictions (`fraud_label_new`) with the dataset's actual analysis (`Fraud_Label`) to measure detection accuracy and pipeline performance.

### 1. Impossible Travel
This feature tracks the velocity required to travel between two consecutive transaction locations for the same user. If the velocity exceeds the physical limits of human travel (e.g., faster than a commercial jet), it strongly indicates fradulent transactions. 

#### Key Implementation Details:
- **Geocoding & Caching:** To convert text-based locations into numerical coordinates efficiently, the pipeline utilizes the **Nominatim API**. It includes a local caching system (`city_coords.csv`) to minimize API call overhead and respect rate limits.
- **Geodesic Distance:** Calculates the actual shortest flight distance (in miles) between the current and previous transaction coordinates using the `geopy.geodesic` library.
- **Proportional Sigmoid Scoring:** Instead of applying a rigid binary cutoff, a **Sigmoid function** is implemented to assign a dynamic risk score between `0.0` and `1.0` based on the calculated velocity ($v$).

#### Mathematical Formula:
$$Risk\ Score = \frac{1}{1 + e^{-k(v - v_{limit})}}$$

- $v = \frac{\text{Distance (miles)}}{\text{Time Difference (hours)}}$
- $v_{limit} = 550\ mph$: The benchmark cruising speed of a commercial jet. At this exact speed, the risk score is exactly `0.5`.
- $k = 0.015$: The sensitivity coefficient that controls the steepness of the risk curve, smoothly accelerating the risk score toward `1.0` as velocity surpasses the limit.

### 2. High Velocity
This feature detects sudden spikes in a user's transaction frequency within a single day. A rapid increase in transaction volume often signals automated bot attacks, card testing, or rapid-fire fraudulent spending before the card is blocked.

#### Key Implementation Details:
- **User-Specific Profiling (Z-Score):** Instead of using a static threshold for transaction counts, the pipeline calculates a personalized **Z-Score** for each user. This measures how many standard deviations the current day's transaction count deviates from the user's historical average.
- **Exception Handling:** To prevent `ZeroDivisionError` or extreme mathematical anomalies for users with static behaviors (where standard deviation is 0 or NaN), the pipeline replaces or fills the standard deviation with a baseline floor value of `0.5`.
- **Domain-Specific Risk Adjustments:** The mapped score undergoes contextual weighting based on the `Merchant_Category` to account for industry-specific fraud profiles:
  - **Electronics (+0.15 Boost):** High-target categories for immediate liquidation are penalized with a risk score boost (capped at `1.0`).
  - **Travel / Restaurants (-0.05 Mitigation):** Common daily spending categories receive a slight risk reduction (floored at `0.0`) to avoid unnecessary customer friction.

#### Mathematical Formula & Mapping:
If Daily_Transaction_Count > avg_daily_count ($\mu$), the Z-Score is calculated as:

$$z = \frac{\text{'Daily\_Transaction\_Count'} - \mu}{\sigma}$$

The Z-Score is then bounded and mapped to a dynamic risk increment between `0.0` and `1.0` using an exponential decay function:

$$Risk\ Score = 1 - e^{-0.5 \times z}$$

- $\mu$: The user's historical mean of daily transaction counts (`avg_daily_count`).
- $\sigma$: The user's standard deviation of daily transaction counts (`std_daily_count`), safely adjusted to prevent division by zero.
