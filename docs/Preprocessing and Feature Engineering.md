## Data Preprocessing and Feature Engineering

The dataset was prepared through several preprocessing and feature engineering steps to improve data quality, transform categorical variables into machine-readable formats, and generate meaningful features for fraud detection analysis.

---

# 1. Data Cleaning

The first stage involved cleaning and validating the raw transaction dataset.

### a) Date Conversion

The `TransactionDate` and `PreviousTransactionDate` columns were converted into datetime format using Pandas.

This was necessary because:

* datetime format allows time-based calculations,
* enables transaction interval analysis,
* supports temporal feature engineering.

```python
df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])
```

---

### b) Removing Extra Whitespace

Categorical columns were cleaned by removing leading and trailing spaces.

This helps:

* avoid duplicate category values caused by formatting inconsistencies,
* improve encoding accuracy.

Example:

* `"ATM "` and `"ATM"` become the same category.

```python
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()
```

---

### c) Removing Invalid Values

Records containing impossible or unrealistic values were removed.

The following conditions were applied:

* negative transaction amounts were removed,
* negative customer ages were removed.

```python
df = df[df["TransactionAmount"] >= 0]
df = df[df["CustomerAge"] >= 0]
```

This improved overall data reliability and reduced noise in the dataset.

---

# 2. Feature Encoding

Machine learning models cannot directly process categorical text variables. Therefore, categorical features were transformed into numerical representations.

---

## a) One-Hot Encoding

Low-cardinality categorical features were transformed using one-hot encoding.

The following features were encoded:

* `TransactionType`
* `Channel`
* `CustomerOccupation`

```python
pd.get_dummies(df, columns=small_cats, drop_first=True)
```

### Why this was done

One-hot encoding:

* converts categories into binary columns,
* prevents models from assuming ordinal relationships,
* improves compatibility with machine learning algorithms.

Example:

| TransactionType | TransactionType_Transfer |
| --------------- | ------------------------ |
| Withdrawal      | 0                        |
| Transfer        | 1                        |

`drop_first=True` was used to avoid multicollinearity and reduce redundant features.

---

## b) Frequency Encoding

High-cardinality categorical variables were encoded using frequency encoding.

The following features were transformed:

* `AccountID`
* `DeviceID`
* `IP Address`
* `MerchantID`
* `Location`

```python
freq = df[col].value_counts()
df[col + "_freq"] = df[col].map(freq)
```

### Why this was done

These columns contain many unique values. Using one-hot encoding on them would:

* create too many columns,
* increase dimensionality,
* reduce model efficiency.

Frequency encoding replaces each category with its occurrence count in the dataset.

Example:

| DeviceID | DeviceID_freq |
| -------- | ------------- |
| D123     | 45            |
| D999     | 2             |

This helps the model capture:

* frequently used devices/accounts,
* rare or suspicious entities.

The original categorical columns were then removed.

```python
df.drop(columns=high_cats, inplace=True)
```

---

# 3. Feature Engineering

Additional features were created to capture transaction behavior patterns that may indicate fraudulent activity.

---

## a) Time Since Previous Transaction

A new feature called `TimeSincePrev` was created by calculating the time difference between the current transaction and the customer's previous transaction.

```python
df["TimeSincePrev"] = (
    df["TransactionDate"] - df["PreviousTransactionDate"]
).dt.total_seconds().fillna(0)
```

### Purpose

This feature helps identify:

* unusually rapid transactions,
* suspicious transaction bursts,
* abnormal account activity patterns.

The value was measured in seconds.

---

## b) Amount-to-Balance Ratio

A feature called `AmountBalanceRatio` was created.

```python
df["AmountBalanceRatio"] = (
    df["TransactionAmount"] / (df["AccountBalance"] + 1)
)
```

### Purpose

This measures how large a transaction is relative to the account balance.

Large ratios may indicate:

* risky withdrawals,
* abnormal spending behavior,
* potential fraud attempts.

`+1` was added to avoid division-by-zero errors.

---

## c) Login Risk Indicator

A binary risk feature called `LoginRisk` was created.

```python
df["LoginRisk"] = df["LoginAttempts"].apply(
    lambda x: 1 if x > 1 else 0
)
```

### Purpose

This feature identifies suspicious login behavior.

* `0` → normal login attempt
* `1` → multiple login attempts detected

Multiple login attempts may indicate:

* unauthorized access attempts,
* credential stuffing,
* account compromise risks.

---

## d) Transaction Velocity

A feature called `TransactionCount` was generated using cumulative transaction counts per account.

```python
df["TransactionCount"] = (
    df.groupby("AccountID").cumcount() + 1
)
```

### Purpose

This feature captures transaction frequency and account activity progression.

High transaction velocity may indicate:

* automated activity,
* fraudulent transaction bursts,
* abnormal user behavior.

---

# 4. Feature Scaling

Finally, numerical features were standardized using `StandardScaler`.

```python
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
```

---

## Why Scaling Was Necessary

The dataset contains features with different ranges.

Examples:

* transaction amount may be in thousands,
* login attempts may range between 0 and 5.

Without scaling:

* larger numerical values could dominate the learning process,
* distance-based algorithms would become biased.

Standardization transforms features to:

* mean = 0
* standard deviation = 1

This improves:

* clustering performance,
* anomaly detection,
* model convergence,
* feature comparability.

---

# Final Features Obtained

After preprocessing and feature engineering, the final dataset included:

### Original Numerical Features

* TransactionAmount
* AccountBalance
* CustomerAge
* LoginAttempts

### Engineered Features

* TimeSincePrev
* AmountBalanceRatio
* LoginRisk
* TransactionCount

### One-Hot Encoded Features

* TransactionType_*
* Channel_*
* CustomerOccupation_*

### Frequency Encoded Features

* AccountID_freq
* DeviceID_freq
* IP Address_freq
* MerchantID_freq
* Location_freq

These transformed features were then used for clustering, anomaly detection, and Bayesian Network modeling in the fraud detection pipeline.
