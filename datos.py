"""
datos.py
========
Descarga y prepara los dibujos del dataset "Quick, Draw!" de Google.

¿Qué es Quick, Draw!?
    Un juego web donde la gente tiene 20 segundos para dibujar algo
    ("gato", "manzana", "bicicleta"...) mientras una IA intenta adivinarlo.
    Google publicó los ~50 millones de dibujos de 345 categorías como dataset
    abierto. Nosotros usamos la versión "numpy_bitmap": cada dibujo ya viene
    convertido a una imagen de 28 x 28 píxeles en escala de grises
    (el mismo tamaño que MNIST), guardada como un vector de 784 números (0-255).

Fuente: https://github.com/googlecreativelab/quickdraw-dataset
"""

import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

URL_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"


def descargar_categoria(categoria: str, carpeta: str = "data") -> str:
    """
    Descarga el archivo <categoria>.npy (solo la primera vez) y devuelve su ruta.
    Ejemplo: 'cat' -> data/cat.npy  (~100 MB, unos 120.000 dibujos de gatos)
    """
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, f"{categoria}.npy")
    if os.path.exists(ruta):
        print(f"[datos] Usando archivo ya descargado: {ruta}")
        return ruta

    # Las categorías con espacios ("smiley face") se codifican en la URL (%20)
    url = URL_BASE + urllib.parse.quote(categoria) + ".npy"
    ruta_temporal = ruta + ".descargando"
    print(f"[datos] Descargando la categoría '{categoria}' desde:\n        {url}")

    def mostrar_progreso(bloques, tam_bloque, total):
        if total > 0:
            pct = min(100, bloques * tam_bloque * 100 // total)
            print(f"\r[datos] {pct:3d}% de {total / 1e6:.1f} MB", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, ruta_temporal, reporthook=mostrar_progreso)
    except urllib.error.HTTPError as e:
        if os.path.exists(ruta_temporal):
            os.remove(ruta_temporal)
        raise SystemExit(
            f"\n[datos] No se encontró la categoría '{categoria}' (HTTP {e.code}).\n"
            "        Revisa el nombre exacto (en inglés) en la lista oficial:\n"
            "        https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt"
        )
    print()
    os.replace(ruta_temporal, ruta)  # solo queda el archivo si la descarga terminó
    return ruta


def cargar_quickdraw(categoria: str, n_imagenes: int, tam_lote: int,
                     semilla: int = 42) -> tuple[DataLoader, torch.Tensor, int]:
    """
    Devuelve:
        cargador   -> DataLoader que entrega lotes de imágenes (lote, 1, 28, 28)
        imagenes   -> tensor con todas las imágenes usadas (para graficar ejemplos)
        total      -> cuántos dibujos tiene la categoría completa
    """
    ruta = descargar_categoria(categoria)

    # mmap_mode="r": no carga los ~100 MB a la RAM, lee del disco solo lo que
    # se necesita. Importante con 8 GB de RAM.
    todos = np.load(ruta, mmap_mode="r")          # forma (N, 784), valores 0-255
    total = todos.shape[0]

    # Tomamos una muestra aleatoria (reproducible por la semilla) para que el
    # entrenamiento sea rápido en un portátil. Más imágenes = mejor, pero más lento.
    n = min(n_imagenes, total)
    rng = np.random.default_rng(semilla)
    indices = np.sort(rng.choice(total, size=n, replace=False))
    x = np.asarray(todos[indices], dtype=np.float32)

    # NORMALIZACIÓN: 0..255  ->  0..1  ->  -1..1
    # Se usa [-1, 1] porque la última capa del generador es Tanh, que produce
    # exactamente ese rango. Reales y falsas deben estar en la misma escala.
    x = x / 255.0
    x = x * 2.0 - 1.0

    # 784 -> 1 x 28 x 28  (canal, alto, ancho), el formato que esperan las capas conv
    x = x.reshape(-1, 1, 28, 28)
    imagenes = torch.from_numpy(x)

    cargador = DataLoader(
        TensorDataset(imagenes),
        batch_size=tam_lote,
        shuffle=True,       # baraja en cada época
        drop_last=True,     # descarta el último lote incompleto (BatchNorm lo agradece)
        num_workers=0,      # 0 = sin procesos extra (lo más seguro en Windows)
    )
    print(f"[datos] Categoría '{categoria}': {total:,} dibujos disponibles, "
          f"usando {n:,} -> {len(cargador)} lotes de {tam_lote} por época")
    return cargador, imagenes, total
