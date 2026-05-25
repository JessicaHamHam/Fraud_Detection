# Sherlock in My Wallet: Fraud Detection System Based on Customer Behavior

A Python-based data pipeline optimized for analyzing customer behavioral patterns to detect fraudulent transactions, using a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset).

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

#### Mathematical Formula:
If `Daily_Transaction_Count` > `avg_daily_count` ($\mu$), the Z-Score is calculated as:

$$z = \frac{\text{Daily Transaction Count} - \mu}{\sigma}$$

The Z-Score is then bounded and mapped to a dynamic risk increment between `0.0` and `1.0` using an exponential decay function:

$$Risk\ Score = 1 - e^{-0.5 \times z}$$

- $\mu$: The user's historical mean of daily transaction counts (`avg_daily_count`).
- $\sigma$: The user's standard deviation of daily transaction counts (`std_daily_count`), safely adjusted to prevent division by zero.

### 3. Small-to-Large Transaction Blast
This feature captures a common fraud pattern known as "Card Testing" or "Credit Blasting." Fraudsters often initiate a low-value transaction to verify if a stolen card is active. Once confirmed, they immediately execute a massive transaction to drain the card's limit before it gets blocked.

#### Key Implementation Details:
- **Amount Explosion Ratio:** Tracks the multiplier gap between the current transaction amount and the immediately preceding one. The pipeline filters out micro-fluctuations and only triggers if the current amount is **at least 10 times larger** ($Ratio \ge 10.0$) than the previous one.
- **Strict Time Bounding:** This logic only monitors rapid-fire sequences occurring **within a strict 1-hour window** ($Time\ Diff < 1.0\ hour$). Any transactions spaced further apart are safely ignored to eliminate normal organic spending behaviors.
- **Exponential Time Decay:** Incorporates a time-decay factor. The shorter the time gap between the small test and the large blast, the more urgent and severe the risk factor becomes.
- **Contextual Category Multiplier:** If the targeted high-amount transaction occurs in high-liquidation sectors like **Electronics** or **Travel**, the final risk increment receives a **30% compounding multiplier ($\times 1.3$)**, capped at a maximum value of `1.0`.

#### Mathematical Formula:
First, the pipeline calculates the scale of the price gap ($Ratio$) between the current and previous transaction amounts:

$$Ratio = \frac{Amount_{current}}{Amount_{prev}}$$

- $Amount_{current}$: Current transaction amount (`Transaction_Amount`)
- $Amount_{prev}$: Previous transaction amount (`prev_amount`)

If $Ratio \ge 10.0$ within 1 hour, the base risk score is mapped via a Sigmoid function and then decayed by time:

$$Base\ Risk = \frac{1}{1 + e^{-k_{amount}(Ratio - Ratio_{limit})}}$$
$$Risk\ Score_{final} = Base\ Risk \times e^{-3.0 \times \Delta t}$$

- $Ratio$: The scale of the price gap ($\text{`Transaction Amount`} / \text{`prev amount`}$).
- $Ratio_{limit} = 50.0$: The midpoint threshold where the base risk curve reaches exactly `0.5` (a 50x increase in transaction size).
- $k_{amount} = 0.05$: The slope sensitivity adjusting how aggressively the risk scales.
- $\Delta t$: The time difference in hours (`time_diff_hours`).

### 4. Account Takeover (ATO) Detection via Behavioral Biometrics
This feature targets Account Takeover (ATO) scenarios where an unauthorized attacker gains access to a legitimate user's account. Attackers typically show sudden behavioral shifts—such as failing credentials multiple times, switching devices, changing authentication methods, or transacting from unusual networks and distances.

#### Key Implementation Details:
- **Baseline User Profiling:** The pipeline creates a personalized profile for each user using historical data:
  - `usual_device`: The user's most frequently used device type (mode).
  - `usual_auth`: The user's most frequently used authentication method (mode).
  - `avg_distance`: The user's historical average transaction distance (mean).
- **The Suspicion Trigger:** The scoring logic triggers if a user has **more than 3 failed transactions within the last 7 days** (`Failed_Transaction_Count_7d > 3`).
- **Multi-Factor Risk Scoring:** Once triggered, risk increments are systematically compounded based on the following indicators:
  - **Indicator A (Authentication Change):** Current authentication method differs from `usual_auth` $\rightarrow$ **+0.20 Risk**
  - **Indicator B (Device Switch):** Current device type differs from `usual_device` $\rightarrow$ **+0.20 Risk**
  - **Indicator C (Network & Location Anomaly):** Evaluates spatial and network shifts based on IP flagging and distance explosion:
    - An anomaly is flagged if `IP_Address_Flag == 1` or if the current transaction distance is **3 times larger** than the user's `avg_distance`.
    - **Single Anomaly** (IP *or* Distance): $\rightarrow$ **+0.20 Risk**
    - **Compound Anomaly** (IP *and* Distance): $\rightarrow$ **+0.50 Risk** 
