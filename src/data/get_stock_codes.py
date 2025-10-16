import requests
import json
import argparse
from pathlib import Path
from src.config import settings


def get_auth_token(base_url, username, password):
    """Gets an authentication token from the API."""
    token_url = f"{base_url}/api/v1/token"
    print(f"1. Attempting to get the access token from: {token_url}")

    try:
        response = requests.post(token_url, auth=(username, password))
        response.raise_for_status()  # Raises an error for status >= 400
        token = response.json().get('token')
        print("   => Token obtained successfully!\n")
        return token
    except requests.exceptions.RequestException as e:
        print(f"   => ERROR obtaining token: {e}")
        return None


def get_premium_stock_data(base_url, token):
    """Fetches data from a protected endpoint using the JWT token."""
    data_url = f"{base_url}/api/v1/stocks/codes"
    print(f"2. Accessing protected endpoint at: {data_url}")

    if not token:
        print("   => ERROR: No token provided.")
        return None

    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(data_url, headers=headers)
        response.raise_for_status()
        print("   => Access granted! Data received successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"   => ERROR accessing data: {e}")
        return None


def save_data_to_json(data, output_path):
    """Saves the data to a JSON file, creating the directory if necessary."""
    try:
        output_file = Path(output_path)
        # Create the parent directory of the file if it does not exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n3. Data saved successfully to: {output_path}")
    except Exception as e:
        print(f"\nERROR saving the file: {e}")
        exit(1)  # Exit with an error code so DVC knows it failed


if __name__ == "__main__":
    """Main function to orchestrate the process."""

    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Fetches stock codes from an API and saves them to JSON.")
    parser.add_argument("--output", required=True, help="Path for the output JSON file.")
    args = parser.parse_args()

    # Get credentials and URL from environment variables
    username = settings.STRATEGIA_INVEST_API_USERNAME
    password = settings.STRATEGIA_INVEST_API_PASSWORD
    base_url = settings.STRATEGIA_INVEST_API_BASE_URL

    if not all([username, password, base_url]):
        print(
            "ERROR: The environment variables API_USERNAME, API_PASSWORD, and API_BASE_URL must be set")
        exit(1)  # Exit with an error code

    # Get the token
    auth_token = get_auth_token(base_url, username, password)
    if not auth_token:
        print("Process interrupted due to failure in obtaining the token.")
        exit(1)

    # Get the data
    stock_data = get_premium_stock_data(base_url, auth_token)
    if not stock_data:
        print("Process interrupted due to failure in obtaining the data.")
        exit(1)

    # Save the data
    save_data_to_json(stock_data, args.output)
    print("\n--- Process completed successfully! ---")