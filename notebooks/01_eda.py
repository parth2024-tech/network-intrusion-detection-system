# NIDS EDA — Exploratory Data Analysis
# ======================================
# Run this notebook after loading your raw CICIDS2017 data.
# Execute cells top-to-bottom in Jupyter.

# %%
import sys

sys.path.insert(0, "..")  # so we can import src.*


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["figure.dpi"] = 130
sns.set_theme(style="whitegrid")

# %%
# === 1. Load raw combined data ===
from src.data.loader import load_cicids2017

df = load_cicids2017("../data/raw")
print(f"Shape: {df.shape}")
df.head(3)

# %%
# === 2. Class distribution ===
from src.data.cleaner import strip_column_names

df = strip_column_names(df)

label_counts = df["Label"].value_counts()
print("Label counts:\n", label_counts.to_string())

fig, ax = plt.subplots(figsize=(12, 4))
label_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Flow Count per Label (CICIDS2017)")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("../reports/plots/eda_class_distribution.png")
plt.show()

# %%
# === 3. Data quality — Inf / NaN counts ===
num_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[num_cols]).sum().sort_values(ascending=False)
nan_counts = df[num_cols].isna().sum().sort_values(ascending=False)

print("Columns with Inf values:")
print(inf_counts[inf_counts > 0])
print("\nColumns with NaN values:")
print(nan_counts[nan_counts > 0])

# %%
# === 4. Feature distributions: Benign vs Attack ===
from src.data.cleaner import encode_binary_labels, impute_missing, replace_infinities

df = replace_infinities(df)
df = impute_missing(df, strategy="median")
df = encode_binary_labels(df, label_col="Label")

key_features = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "SYN Flag Count", "Flow Bytes/s", "Flow Packets/s",
    "Fwd Packet Length Mean",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    if feat not in df.columns:
        axes[i].set_visible(False)
        continue
    benign = df.loc[df["label_binary"] == 0, feat].clip(upper=df[feat].quantile(0.99))
    attack = df.loc[df["label_binary"] == 1, feat].clip(upper=df[feat].quantile(0.99))
    axes[i].hist(benign, bins=50, alpha=0.6, label="Benign", color="green", density=True)
    axes[i].hist(attack, bins=50, alpha=0.6, label="Attack", color="red", density=True)
    axes[i].set_title(feat, fontsize=9)
    axes[i].legend(fontsize=7)

if len(key_features) < len(axes):
    axes[-1].set_visible(False)

plt.suptitle("Feature Distributions: Benign vs Attack", fontsize=13)
plt.tight_layout()
plt.savefig("../reports/plots/eda_feature_distributions.png")
plt.show()

# %%
# === 5. Correlation heatmap (sample 20 features) ===
num_cols_clean = [c for c in num_cols if c in df.columns and c != "label_binary"]
sample_cols = num_cols_clean[:20]

corr = df[sample_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            annot=False, fmt=".1f", linewidths=0.3, ax=ax)
ax.set_title("Feature Correlation (top 20 numeric features)")
plt.tight_layout()
plt.savefig("../reports/plots/eda_correlation_heatmap.png")
plt.show()

# %%
# === 6. Attack type breakdown per source file (day) ===
day_attack = df.groupby(["source_file", "Label"]).size().unstack(fill_value=0)
print("\nAttacks per day:\n", day_attack.to_string())

fig, ax = plt.subplots(figsize=(14, 6))
day_attack.drop(columns="BENIGN", errors="ignore").plot(kind="bar", stacked=True, ax=ax)
ax.set_title("Attack Types per Day (CICIDS2017)")
ax.set_xlabel("Source File")
ax.set_ylabel("Flow Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("../reports/plots/eda_attacks_per_day.png")
plt.show()

print("\n=== EDA complete. Plots saved to ../reports/plots/ ===")
