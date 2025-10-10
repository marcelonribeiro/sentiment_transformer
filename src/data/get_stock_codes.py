import requests
import json
import argparse
from pathlib import Path
from src.config import settings


def get_auth_token(base_url, username, password):
    token_url = f"{base_url}/api/v1/token"
    print(f"1. Tentando obter o token de acesso em: {token_url}")

    try:
        response = requests.post(token_url, auth=(username, password))
        response.raise_for_status()  # Lança um erro para status >= 400
        token = response.json().get('token')
        print("   => Token obtido com sucesso!\n")
        return token
    except requests.exceptions.RequestException as e:
        print(f"   => ERRO ao obter token: {e}")
        return None


def get_premium_stock_data(base_url, token):
    """Busca os dados de um endpoint protegido usando o token JWT."""
    data_url = f"{base_url}/api/v1/stocks/codes"
    print(f"2. Acessando endpoint protegido em: {data_url}")

    if not token:
        print("   => ERRO: Nenhum token fornecido.")
        return None

    headers = {'Authorization': f'Bearer {token}'}
    try:
        response = requests.get(data_url, headers=headers)
        response.raise_for_status()
        print("   => Acesso permitido! Dados recebidos com sucesso.")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"   => ERRO ao acessar dados: {e}")
        return None


def save_data_to_json(data, output_path):
    """Salva os dados em um arquivo JSON, criando o diretório se necessário."""
    try:
        output_file = Path(output_path)
        # Cria o diretório pai do arquivo, se ele não existir
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n3. Dados salvos com sucesso em: {output_path}")
    except Exception as e:
        print(f"\nERRO ao salvar o arquivo: {e}")
        exit(1)  # Sai com código de erro para o DVC saber que falhou


if __name__ == "__main__":
    """Função principal para orquestrar o processo."""

    # Configura os argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Busca códigos de ações de uma API e salva em JSON.")
    parser.add_argument("--output", required=True, help="Caminho do arquivo JSON de saída.")
    args = parser.parse_args()

    # Pega as credenciais e a URL das variáveis de ambiente
    username = settings.STRATEGIA_INVEST_API_USERNAME
    password = settings.STRATEGIA_INVEST_API_PASSWORD
    base_url = settings.STRATEGIA_INVEST_API_BASE_URL

    if not all([username, password, base_url]):
        print(
            "ERRO: As variáveis de ambiente API_USERNAME, API_PASSWORD e API_BASE_URL devem estar definidas no arquivo .env")
        exit(1)  # Sai com código de erro

    # Passo 1: Obter o token
    auth_token = get_auth_token(base_url, username, password)
    if not auth_token:
        print("Processo interrompido devido à falha na obtenção do token.")
        exit(1)

    # Passo 2: Obter os dados
    stock_data = get_premium_stock_data(base_url, auth_token)
    if not stock_data:
        print("Processo interrompido devido à falha na obtenção dos dados.")
        exit(1)

    # Passo 3: Salvar os dados
    save_data_to_json(stock_data, args.output)
    print("\n--- Processo concluído com sucesso! ---")
