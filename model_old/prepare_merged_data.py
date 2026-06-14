#!/usr/bin/env python3
"""
Data preparation script for merging old and new speaking assessment data.
Performs stratified splitting and rebalancing to flatten score distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from collections import Counter

# Paths
OLD_DATA_DIR = "/home/user06/data/Speaking_VSTEP/Label/data_groupby_candidateID"
NEW_DATA_PATH = "/home/user06/data/Speaking_VSTEP/Test_V2/grouped_by_candidate.csv"
OUTPUT_DIR = "/home/user06/data/Speaking_VSTEP/Test_V2"

# Score bin definitions
# Adjusted: 4.0 is now in MIDDLE category for downsampling
EDGE_SCORES = list(np.arange(0.0, 4.0, 0.5)) + list(np.arange(7.0, 10.5, 0.5))
MIDDLE_SCORES = list(np.arange(4.0, 7.0, 0.5))

def load_old_data():
    """Load and merge ALL old data (train+val+test) into one dataset."""
    print("Loading old data...")
    old_train = pd.read_csv(os.path.join(OLD_DATA_DIR, "train_data_grouped_by_candidateID.csv"))
    old_val = pd.read_csv(os.path.join(OLD_DATA_DIR, "val_data_grouped_by_candidateID.csv"))
    old_test = pd.read_csv(os.path.join(OLD_DATA_DIR, "test_data_grouped_by_candidateID.csv"))

    old_train['source'] = 'old'
    old_val['source'] = 'old'
    old_test['source'] = 'old'

    print(f"Old data - Train: {len(old_train)}, Val: {len(old_val)}, Test: {len(old_test)}")

    # Merge ALL old data together
    old_combined = pd.concat([old_train, old_val, old_test], ignore_index=True)
    print(f"Old data combined (all splits merged): {len(old_combined)} samples")

    return old_combined

def split_new_data():
    """Load and split new data with stratification on 'grammar' score."""
    print("\nLoading and splitting new data...")
    new_data = pd.read_csv(NEW_DATA_PATH)
    new_data['source'] = 'new'

    # Round scores to nearest 0.5 for stratification
    new_data['grammar_rounded'] = (new_data['grammar'] * 2).round() / 2

    # Check class distribution
    class_counts = new_data['grammar_rounded'].value_counts()
    print(f"Score distribution in new data:")
    print(class_counts.sort_index())

    # Identify classes with enough samples for stratified split
    # Need at least 10 samples per class for 80-10-10 split
    min_samples_for_stratify = 10
    stratifiable_classes = class_counts[class_counts >= min_samples_for_stratify].index.tolist()
    non_stratifiable_classes = class_counts[class_counts < min_samples_for_stratify].index.tolist()

    if non_stratifiable_classes:
        print(f"\nWarning: Found {len(non_stratifiable_classes)} score bins with <{min_samples_for_stratify} samples:")
        print(f"Scores: {sorted(non_stratifiable_classes)}")
        print(f"Total samples in rare bins: {sum(class_counts[non_stratifiable_classes])}")
        print("These will be added to training set without stratification.")

        # Separate stratifiable and non-stratifiable data
        stratifiable_mask = new_data['grammar_rounded'].isin(stratifiable_classes)
        stratifiable_data = new_data[stratifiable_mask].copy()
        rare_data = new_data[~stratifiable_mask].copy()

        if len(stratifiable_data) > 0:
            # Split stratifiable data with stratification
            train_strat, temp_strat = train_test_split(
                stratifiable_data,
                test_size=0.2,
                stratify=stratifiable_data['grammar_rounded'],
                random_state=42
            )

            # Check if temp has enough samples for second stratified split
            temp_counts = temp_strat['grammar_rounded'].value_counts()
            if temp_counts.min() >= 2:
                val_strat, test_strat = train_test_split(
                    temp_strat,
                    test_size=0.5,
                    stratify=temp_strat['grammar_rounded'],
                    random_state=42
                )
            else:
                # Can't stratify second split, just split randomly
                print("Warning: Second split cannot be stratified, using random split")
                val_strat, test_strat = train_test_split(
                    temp_strat,
                    test_size=0.5,
                    random_state=42
                )

            # Add rare data to train set
            train_new = pd.concat([train_strat, rare_data], ignore_index=True)
            val_new = val_strat
            test_new = test_strat

            print(f"Added {len(rare_data)} rare samples to training set")

        else:
            # All data is rare, just split randomly
            print("Warning: All data is rare, using random split")
            train_new, temp = train_test_split(new_data, test_size=0.2, random_state=42)
            val_new, test_new = train_test_split(temp, test_size=0.5, random_state=42)

    else:
        # All classes have enough samples for stratified split
        train_new, temp = train_test_split(
            new_data,
            test_size=0.2,
            stratify=new_data['grammar_rounded'],
            random_state=42
        )

        val_new, test_new = train_test_split(
            temp,
            test_size=0.5,
            stratify=temp['grammar_rounded'],
            random_state=42
        )

    # Remove helper column
    train_new = train_new.drop(columns=['grammar_rounded'])
    val_new = val_new.drop(columns=['grammar_rounded'])
    test_new = test_new.drop(columns=['grammar_rounded'])

    print(f"New data - Train: {len(train_new)}, Val: {len(val_new)}, Test: {len(test_new)}")
    return train_new, val_new, test_new

def plot_distribution(df, title, filename, bins=None):
    """Plot score distribution histogram."""
    if bins is None:
        bins = np.arange(0, 11, 0.5)

    plt.figure(figsize=(12, 6))
    counts, edges, patches = plt.hist(df['grammar'], bins=bins, edgecolor='black', alpha=0.7)

    # Color code: edge scores (blue), middle scores (orange)
    for i, patch in enumerate(patches):
        score = edges[i]
        if score in EDGE_SCORES:
            patch.set_facecolor('steelblue')
        elif score in MIDDLE_SCORES:
            patch.set_facecolor('coral')

    plt.xlabel('Final Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(np.arange(0, 11, 0.5), rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Edge [0-3.5, 7.0-10]'),
        Patch(facecolor='coral', label='Middle [4.0-6.5]')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot: {filename}")
    plt.close()

def print_distribution_stats(df, label):
    """Print detailed distribution statistics."""
    print(f"\n{'='*60}")
    print(f"{label} - Distribution Statistics")
    print(f"{'='*60}")

    # Round to nearest 0.5
    scores_rounded = (df['grammar'] * 2).round() / 2
    score_counts = Counter(scores_rounded)

    total = len(df)
    print(f"Total samples: {total}")
    print(f"\nScore\tCount\t%\tCategory")
    print("-" * 50)

    for score in np.arange(0, 10.5, 0.5):
        count = score_counts.get(score, 0)
        pct = (count / total * 100) if total > 0 else 0
        category = "EDGE" if score in EDGE_SCORES else "MID"
        print(f"{score:.1f}\t{count}\t{pct:.1f}%\t{category}")

    edge_total = sum(score_counts.get(s, 0) for s in EDGE_SCORES)
    mid_total = sum(score_counts.get(s, 0) for s in MIDDLE_SCORES)
    print("-" * 50)
    print(f"Edge total: {edge_total} ({edge_total/total*100:.1f}%)")
    print(f"Middle total: {mid_total} ({mid_total/total*100:.1f}%)")

def rebalance_training_data(df, target_count=None):
    """
    Rebalance training data by:
    1. Downsampling middle scores [4.5-6.5] from old data only
    2. Upsampling edge scores [0-4.0, 7.0-10] via duplication
    """
    print("\n" + "="*60)
    print("REBALANCING TRAINING DATA")
    print("="*60)

    # Round scores to nearest 0.5
    df['grammar_rounded'] = (df['grammar'] * 2).round() / 2

    # Calculate distribution
    score_counts = Counter(df['grammar_rounded'])

    # Determine target count if not specified
    if target_count is None:
        edge_counts = [score_counts.get(s, 0) for s in EDGE_SCORES if score_counts.get(s, 0) > 0]
        mid_counts = [score_counts.get(s, 0) for s in MIDDLE_SCORES if score_counts.get(s, 0) > 0]

        # Use 70th percentile of edge counts as target
        if edge_counts:
            target_count = int(np.percentile(edge_counts, 70))
        else:
            target_count = int(np.median(mid_counts) * 0.3)  # Fallback

        # Ensure reasonable bounds
        target_count = max(50, min(target_count, 100))

    print(f"Target count per bin: {target_count}")

    balanced_dfs = []

    for score in np.arange(0, 10.5, 0.5):
        score_data = df[df['grammar_rounded'] == score].copy()
        current_count = len(score_data)

        if current_count == 0:
            print(f"Score {score:.1f}: 0 samples (skipping)")
            continue

        if score in MIDDLE_SCORES:
            # Downsample middle scores (old data only, keep all new data)
            old_data = score_data[score_data['source'] == 'old']
            new_data = score_data[score_data['source'] == 'new']

            new_count = len(new_data)
            old_count = len(old_data)

            # Calculate how many old samples to keep
            old_to_keep = max(0, target_count - new_count)

            if old_count > old_to_keep:
                # Downsample old data
                old_sampled = old_data.sample(n=old_to_keep, random_state=42)
                combined = pd.concat([new_data, old_sampled], ignore_index=True)
                print(f"Score {score:.1f}: {current_count} → {len(combined)} (kept {new_count} new + {old_to_keep}/{old_count} old)")
            else:
                # Keep all old data
                combined = score_data
                print(f"Score {score:.1f}: {current_count} (kept all: {new_count} new + {old_count} old)")

            balanced_dfs.append(combined)

        elif score in EDGE_SCORES:
            # Upsample edge scores via duplication
            if current_count < target_count:
                # Calculate how many times to repeat
                n_repeats = target_count // current_count
                remainder = target_count % current_count

                # Repeat entire dataset n_repeats times
                upsampled = pd.concat([score_data] * n_repeats, ignore_index=True)

                # Add random samples for remainder
                if remainder > 0:
                    extra = score_data.sample(n=remainder, replace=True, random_state=42)
                    upsampled = pd.concat([upsampled, extra], ignore_index=True)

                print(f"Score {score:.1f}: {current_count} → {len(upsampled)} (upsampled {len(upsampled)/current_count:.1f}x)")
                balanced_dfs.append(upsampled)
            else:
                # Already at or above target
                print(f"Score {score:.1f}: {current_count} (no upsampling needed)")
                balanced_dfs.append(score_data)

        else:
            # Should not happen
            balanced_dfs.append(score_data)

    # Combine all balanced bins
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Remove helper column
    balanced_df = balanced_df.drop(columns=['grammar_rounded'])

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal samples after balancing: {len(balanced_df)}")
    print(f"Old data: {len(balanced_df[balanced_df['source']=='old'])}")
    print(f"New data: {len(balanced_df[balanced_df['source']=='new'])}")

    return balanced_df

def main():
    print("="*60)
    print("DATA PREPARATION FOR GRAMMAR SCORE TRAINING")
    print("="*60)

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load ALL old data (merged train+val+test)
    old_combined = load_old_data()

    # Step 2: Split new data into 8-1-1
    train_new, val_new, test_new = split_new_data()

    # Step 3: Merge old (all) with train_new ONLY
    # Val and test are from new data only
    print("\nMerging data...")
    train_raw = pd.concat([old_combined, train_new], ignore_index=True)
    val_v2 = val_new  # Only new data
    test_v2 = test_new  # Only new data

    print(f"Train: {len(train_raw)} samples (Old combined: {len(old_combined)}, New train: {len(train_new)})")
    print(f"Val: {len(val_v2)} samples (New only)")
    print(f"Test: {len(test_v2)} samples (New only)")

    # Print distribution before balancing (train only)
    print_distribution_stats(train_raw, "TRAIN SET - BEFORE BALANCING")

    # Step 4: Save unbalanced datasets
    train_v2_path = os.path.join(OUTPUT_DIR, "train_V2.csv")
    val_v2_path = os.path.join(OUTPUT_DIR, "val_V2.csv")
    test_v2_path = os.path.join(OUTPUT_DIR, "test_V2.csv")

    train_raw.to_csv(train_v2_path, index=False)
    val_v2.to_csv(val_v2_path, index=False)
    test_v2.to_csv(test_v2_path, index=False)

    print(f"\nSaved unbalanced datasets:")
    print(f"  {train_v2_path} ({len(train_raw)} samples - Old all + New train)")
    print(f"  {val_v2_path} ({len(val_v2)} samples - New only)")
    print(f"  {test_v2_path} ({len(test_v2)} samples - New only)")

    # Step 5: Plot distribution before balancing
    plot_distribution(
        train_raw,
        "Training Data Distribution - BEFORE Balancing",
        os.path.join("/home/user06/Interspeech_2026/model_old", "distribution_before_balance.png")
    )

    # Step 6: Rebalance training data (only train, keep val/test unbalanced)
    train_balanced = rebalance_training_data(train_raw, target_count=None)

    # Print distribution after balancing
    print_distribution_stats(train_balanced, "TRAIN SET - AFTER BALANCING")

    # Step 7: Save balanced dataset
    train_v2_balance_path = os.path.join(OUTPUT_DIR, "train_V2_balance.csv")
    train_balanced.to_csv(train_v2_balance_path, index=False)
    print(f"\nSaved balanced dataset:")
    print(f"  {train_v2_balance_path}")

    # Step 8: Plot distribution after balancing
    plot_distribution(
        train_balanced,
        "Training Data Distribution - AFTER Balancing",
        os.path.join("/home/user06/Interspeech_2026/model_old", "distribution_after_balance.png")
    )

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  1. {train_v2_path}")
    print(f"  2. {train_v2_balance_path}")
    print(f"  3. {val_v2_path}")
    print(f"  4. {test_v2_path}")
    print(f"\nGenerated plots:")
    print(f"  1. distribution_before_balance.png")
    print(f"  2. distribution_after_balance.png")
    print("\nNext step: Run train_W2VAudio_bycandidates_final_V2.py")

if __name__ == "__main__":
    main()
