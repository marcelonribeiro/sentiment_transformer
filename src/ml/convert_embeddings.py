import argparse
from pathlib import Path
from gensim.models import KeyedVectors

def convert_embeddings_to_binary(input_path, output_path):
    """
    Carrega um arquivo de word embeddings no formato .vec e o salva no
    formato binário nativo do Gensim para um carregamento mais rápido.
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    # 1. Validar se o arquivo de entrada existe
    if not input_file.is_file():
        print(f"ERRO: Arquivo de entrada não encontrado em '{input_file}'")
        exit(1) # Sai com código de erro para o DVC

    # 2. Criar o diretório de saída se ele não existir
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Diretório de saída '{output_file.parent}' garantido.")

    print(f"Iniciando conversão de '{input_file}' (isso pode levar vários minutos)...")

    # 3. Carregar o arquivo de texto (a parte lenta)
    try:
        word_vectors = KeyedVectors.load_word2vec_format(input_file)
    except Exception as e:
        print(f"ERRO ao carregar o arquivo de embeddings: {e}")
        exit(1)

    # 4. Salvar os vetores no formato binário (rápido)
    # O Gensim automaticamente cria o arquivo principal .bin e o arquivo .npy associado.
    try:
        word_vectors.save(str(output_file))
    except Exception as e:
        print(f"ERRO ao salvar o arquivo binário: {e}")
        exit(1)

    print(f"\nConversão completa! Arquivos binários salvos com base em: '{output_file}'")

if __name__ == "__main__":
    """
    Função principal para parsear argumentos e iniciar a conversão.
    """
    parser = argparse.ArgumentParser(
        description="Converte embeddings de texto (.vec) para o formato binário (.bin) do Gensim."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Caminho para o arquivo de entrada .vec."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Caminho para o arquivo de saída .bin. O arquivo .npy correspondente será criado automaticamente."
    )
    args = parser.parse_args()

    convert_embeddings_to_binary(args.input, args.output)
