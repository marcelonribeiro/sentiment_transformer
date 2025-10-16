import argparse
from pathlib import Path
from gensim.models import KeyedVectors

def convert_embeddings_to_binary(input_path, output_path):
    """
    Loads a word embeddings file in .vec format and saves it in
    Gensim's native binary format for faster loading.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    # Validate if the input file exists
    if not input_file.is_file():
        print(f"ERROR: Input file not found at '{input_file}'")
        exit(1) # Exit with an error code for DVC

    # Create the output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output directory '{output_file.parent}' ensured.")

    print(f"Starting conversion of '{input_file}' (this may take several minutes)...")

    # Load the text file (the slow part)
    try:
        word_vectors = KeyedVectors.load_word2vec_format(input_file)
    except Exception as e:
        print(f"ERROR loading the embeddings file: {e}")
        exit(1)

    # Save the vectors in binary format (fast)
    # Gensim automatically creates the main .bin file and the associated .npy file.
    try:
        word_vectors.save(str(output_file))
    except Exception as e:
        print(f"ERROR saving the binary file: {e}")
        exit(1)

    print(f"\nConversion complete! Binary files saved based on: '{output_file}'")

if __name__ == "__main__":
    """
    Main function to parse arguments and start the conversion.
    """
    parser = argparse.ArgumentParser(
        description="Converts text embeddings (.vec) to Gensim's binary format (.bin)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input .vec file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output .bin file. The corresponding .npy file will be created automatically."
    )
    args = parser.parse_args()

    convert_embeddings_to_binary(args.input, args.output)