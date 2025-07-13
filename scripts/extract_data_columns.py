import argparse
import csv
import os
import sys


def read_tsv_columns(file_path):
    """Read the column headers from a TSV file."""
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            columns = next(reader)  # Get the first row (headers)
            return columns
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except (IOError, UnicodeDecodeError, csv.Error) as e:
        print(f"Error reading columns from '{file_path}': {e}")
        sys.exit(1)


def read_tsv_data(file_path, target_columns):
    """Read data from a TSV file for specific columns."""
    try:
        with open(file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")

            # Check if all target columns exist in the source file
            missing_columns = [col for col in target_columns if col not in reader.fieldnames]
            if missing_columns:
                print(f"Error: Columns {missing_columns} not found in '{file_path}'")
                print(f"Available columns: {reader.fieldnames}")
                sys.exit(1)

            # Extract data for target columns
            data = []
            for row in reader:
                extracted_row = {col: row[col] for col in target_columns}
                data.append(extracted_row)

            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except (IOError, UnicodeDecodeError, csv.Error) as e:
        print(f"Error reading data from '{file_path}': {e}")
        sys.exit(1)


def write_tsv_data(output_path, columns, data):
    """Write data to a TSV file."""
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            writer.writerows(data)
        print(f"Successfully created output file: {output_path}")
    except (IOError, UnicodeDecodeError, csv.Error) as e:
        print(f"Error writing to '{output_path}': {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract columns from one TSV file and populate with data from another TSV file"
    )

    parser.add_argument(
        "--columns-file", required=True, help="Path to TSV file containing the desired column structure"
    )

    parser.add_argument("--data-file", required=True, help="Path to TSV file containing the data to extract")

    parser.add_argument("--output-file", required=True, help="Path for the output TSV file")

    parser.add_argument("--data-dir", default=".", help="Base directory for data files (default: current directory)")

    args = parser.parse_args()

    # Resolve file paths relative to data directory if they're not absolute
    columns_file = (
        args.columns_file if os.path.isabs(args.columns_file) else os.path.join(args.data_dir, args.columns_file)
    )
    data_file = args.data_file if os.path.isabs(args.data_file) else os.path.join(args.data_dir, args.data_file)
    output_file = (
        args.output_file if os.path.isabs(args.output_file) else os.path.join(args.data_dir, args.output_file)
    )

    print(f"Reading column structure from: {columns_file}")
    print(f"Reading data from: {data_file}")
    print(f"Writing output to: {output_file}")

    # Step 1: Get columns from the first file
    target_columns = read_tsv_columns(columns_file)
    print(f"Target columns ({len(target_columns)}): {target_columns[:5]}{'...' if len(target_columns) > 5 else ''}")

    # Step 2: Extract data for those columns from the second file
    extracted_data = read_tsv_data(data_file, target_columns)
    print(f"Extracted {len(extracted_data)} rows of data")

    # Step 3: Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 4: Write the new TSV file
    write_tsv_data(output_file, target_columns, extracted_data)


if __name__ == "__main__":
    main()
