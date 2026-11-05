"""
visualizacion.py
================
Gráficas e imágenes de resultados:
    - cuadrículas de dibujos (reales y generados)
    - curvas de las funciones de pérdida
    - probabilidades D(x) y D(G(z)) por época
    - histogramas de los pesos de cada red
    - GIF con la evolución del generador
"""

import os

import matplotlib
matplotlib.use("Agg")   # guarda imágenes sin abrir ventanas (no bloquea la terminal)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLOR_D = "#c2410c"   # discriminador: naranja
COLOR_G = "#1d4ed8"   # generador: azul


def _a_numpy(x):
    """Convierte un tensor de PyTorch (o array) a numpy en la CPU."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def guardar_cuadricula(imagenes, ruta: str, titulo: str, columnas: int = 8) -> None:
    """
    imagenes: (n, 1, 28, 28) con valores en [-1, 1]
    Arma un mosaico y lo guarda. Se dibuja con trazo negro sobre fondo blanco,
    como se ven los dibujos en el juego.
    """
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    x = _a_numpy(imagenes)[:, 0]                 # (n, 28, 28)
    x = (np.clip(x, -1, 1) + 1) / 2              # [-1,1] -> [0,1]
    n = x.shape[0]
    filas = int(np.ceil(n / columnas))
    sep = 2                                      # separación entre dibujos (px)
    lado = 28 + sep
    mosaico = np.full((filas * lado + sep, columnas * lado + sep), 0.12)
    for i in range(n):
        f, c = divmod(i, columnas)
        mosaico[sep + f * lado: sep + f * lado + 28,
                sep + c * lado: sep + c * lado + 28] = x[i]

    fig = plt.figure(figsize=(7, 7.6), dpi=110)
    plt.imshow(mosaico, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    plt.title(titulo, fontsize=10)
    plt.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.94)
    fig.savefig(ruta)       # sin bbox_inches="tight": todas del mismo tamaño (para el GIF)
    plt.close(fig)


def _media_movil(valores, ventana: int = 50):
    v = np.asarray(valores, dtype=float)
    if len(v) < ventana:
        return v
    return np.convolve(v, np.ones(ventana) / ventana, mode="valid")


def graficar_perdidas(perdidas_iter_D, perdidas_iter_G, historial, ruta, categoria):
    """Curvas de las funciones de pérdida: por lote (suavizada) y por época."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)

    ax1.plot(perdidas_iter_D, color=COLOR_D, alpha=0.15, linewidth=0.8)
    ax1.plot(perdidas_iter_G, color=COLOR_G, alpha=0.15, linewidth=0.8)
    ax1.plot(_media_movil(perdidas_iter_D), color=COLOR_D, label="Discriminador (L_D)")
    ax1.plot(_media_movil(perdidas_iter_G), color=COLOR_G, label="Generador (L_G)")
    ax1.set_title("Pérdida por lote (línea fuerte = media móvil de 50)")
    ax1.set_xlabel("Iteración (lote)")
    ax1.set_ylabel("Pérdida BCE")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ep = [h["epoca"] for h in historial]
    ax2.plot(ep, [h["perdida_D"] for h in historial], "o-", color=COLOR_D, label="L_D total")
    ax2.plot(ep, [h["perdida_D_real"] for h in historial], "--", color=COLOR_D, alpha=0.6,
             label="L_D parte reales")
    ax2.plot(ep, [h["perdida_D_falsa"] for h in historial], ":", color=COLOR_D, alpha=0.8,
             label="L_D parte falsas")
    ax2.plot(ep, [h["perdida_G"] for h in historial], "o-", color=COLOR_G, label="L_G")
    ax2.set_title("Pérdida promedio por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Pérdida BCE")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(f"Funciones de pérdida — GAN Quick, Draw! — categoría '{categoria}'")
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def graficar_probabilidades(historial, ruta, categoria):
    """D(x) y D(G(z)) por época. En el equilibrio teórico ambas tienden a 0.5."""
    ep = [h["epoca"] for h in historial]
    fig = plt.figure(figsize=(8, 4.6), dpi=110)
    plt.plot(ep, [h["D_x"] for h in historial], "o-", color="#15803d",
             label="D(x): prob. de 'real' a dibujos REALES")
    plt.plot(ep, [h["D_G_z"] for h in historial], "o-", color="#7e22ce",
             label="D(G(z)): prob. de 'real' a dibujos FALSOS")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1,
                label="0.5 = D ya no distingue (equilibrio ideal)")
    plt.ylim(0, 1)
    plt.xlabel("Época")
    plt.ylabel("Probabilidad promedio")
    plt.title(f"Salidas del discriminador — categoría '{categoria}'")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def histogramas_pesos(modelo, ruta, titulo):
    """Un histograma por cada matriz de pesos (densas y convoluciones) de la red."""
    capas = [(n.replace("red.", ""), _a_numpy(p).ravel())
             for n, p in modelo.named_parameters()
             if n.endswith("weight") and p.dim() > 1]
    fig, ejes = plt.subplots(1, len(capas), figsize=(4.2 * len(capas), 3.6), dpi=110)
    ejes = np.atleast_1d(ejes)
    for ax, (nombre, valores) in zip(ejes, capas):
        ax.hist(valores, bins=80, color="#475569")
        ax.set_title(f"{nombre}\n{valores.size:,} pesos | media {valores.mean():.3f} "
                     f"| desv {valores.std():.3f}", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def crear_gif(rutas_imagenes, ruta_gif, ms_por_cuadro: int = 450) -> None:
    """Une las cuadrículas de cada época en un GIF animado."""
    cuadros = [Image.open(r).convert("P", palette=Image.Palette.ADAPTIVE) for r in rutas_imagenes]
    if not cuadros:
        return
    cuadros[0].save(ruta_gif, save_all=True, append_images=cuadros[1:],
                    duration=ms_por_cuadro, loop=0)
