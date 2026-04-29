"""
Módulo de download e extração do dataset Olist do Kaggle.

Realiza o download do arquivo ZIP, extrai os CSVs e valida a integridade
dos arquivos esperados.
"""

import os
import time
import zipfile

import requests
from tqdm import tqdm

from src.config import DATA_DIR, DATASET_FILES, KAGGLE_DATASET_URL


def download_dataset() -> None:
    """Baixa o dataset do Kaggle com barra de progresso."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "brazilian-ecommerce.zip")

    # Verifica se todos os CSVs já existem
    existing = [
        f for f in DATASET_FILES if os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if len(existing) == len(DATASET_FILES):
        print("[download] Todos os arquivos CSV já existem. Pulando download.")
        return

    print("[download] Baixando dataset do Kaggle...")
    response = requests.get(KAGGLE_DATASET_URL, stream=True, timeout=120)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(zip_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc="Download") as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print("[download] Download concluído. Extraindo arquivos...")
    _extract_and_cleanup(zip_path)


def _extract_and_cleanup(zip_path: str) -> None:
    """Extrai o ZIP e remove o arquivo compactado."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    time.sleep(0.5)
    try:
        os.remove(zip_path)
        print("[download] Arquivo ZIP removido.")
    except OSError as e:
        print(f"[download] Aviso: não foi possível remover o ZIP: {e}")

    _validate_files()


def _validate_files() -> None:
    """Valida que todos os CSVs esperados existem após extração."""
    missing = [
        f for f in DATASET_FILES if not os.path.exists(os.path.join(DATA_DIR, f))
    ]
    if missing:
        raise FileNotFoundError(
            f"[download] Arquivos ausentes após extração: {missing}"
        )
    print(f"[download] {len(DATASET_FILES)} arquivos CSV validados com sucesso.")
