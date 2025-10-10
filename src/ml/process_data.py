import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def process_data(input_csv: Path, output_dir: Path):
    """
    Reads the raw dataset, cleans, balances, and splits it into
    train, validation, and test sets.
    """
    print("Starting data processing...")

    # Load and Clean Data
    try:
        df = pd.read_csv(input_csv, dtype={'product_id': str, 'user_id': str})
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_csv}'")
        exit(1)

    df = df[['review_text', 'overall_rating']].dropna()
    df.columns = ['text', 'rating']

    df_negative = df.loc[df['rating'] < 3].copy()
    df_negative['sentiment'] = 0
    df_positive = df.loc[df['rating'] == 5].copy()
    df_positive['sentiment'] = 1

    df_filtered = pd.concat([df_negative, df_positive])

    # Balance Dataset
    n_samples = df_filtered.groupby('sentiment')['rating'].count().min()
    print(f"Balancing dataset with {n_samples} samples per class.")
    df_neg_balanced = df_negative.sample(n=n_samples, random_state=42)
    df_pos_balanced = df_positive.sample(n=n_samples, random_state=42)
    df_balanced = pd.concat([df_neg_balanced, df_pos_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_balanced.dropna(subset=['text'], inplace=True)
    df_balanced['text'] = df_balanced['text'].astype(str)

    text_clean = df_balanced['text'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df_final = df_balanced[text_clean.str.strip().astype(bool)].copy()
    print(f"Final dataset for splitting has {len(df_final)} samples.")

    # Split into Train, Validation, and Test
    train_df, temp_df = train_test_split(
        df_final,
        test_size=0.3,
        random_state=42,
        stratify=df_final['sentiment']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['sentiment']
    )
    print(f"Data split: Train({len(train_df)}), Validation({len(val_df)}), Test({len(test_df)})")

    # Save Files
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Files saved successfully in '{output_dir}'.")


if __name__ == "__main__":
    """Main function to parse arguments and run the data processing."""
    parser = argparse.ArgumentParser(description="Clean, balance, and split the raw review data.")

    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the raw input CSV file (e.g., B2W-Reviews01.csv)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the train.csv, val.csv, and test.csv files."
    )

    args = parser.parse_args()

    process_data(input_csv=Path(args.input_csv), output_dir=Path(args.output_dir))
