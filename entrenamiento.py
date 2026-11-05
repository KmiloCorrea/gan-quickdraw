"""
entrenamiento.py
================
FUNCIONES DE PÉRDIDA y BUCLE DE ENTRENAMIENTO de la GAN.

La idea (juego de dos jugadores):
    - El DISCRIMINADOR (D) quiere acertar:  reales -> 1,  falsas -> 0
    - El GENERADOR    (G) quiere engañarlo: que D diga 1 a sus falsas

Ambas pérdidas usan ENTROPÍA CRUZADA BINARIA (BCE):

    BCE(p, y) = -[ y * log(p) + (1 - y) * log(1 - p) ]

    donde p = probabilidad que da D (0 a 1) y y = etiqueta objetivo (1 o 0).
    Si D acierta con seguridad, la BCE es ~0. Si se equivoca con seguridad, crece mucho.

Esto implementa la pérdida "minimax" del paper original de Goodfellow (2014),
con la modificación que recomienda el propio paper (y el curso de Google)
para el generador: en vez de MINIMIZAR log(1 - D(G(z))), MAXIMIZA log(D(G(z))).
Al inicio D rechaza fácilmente las falsas, y la versión original da
gradientes casi nulos (G no aprende); la versión modificada no tiene ese problema.
"""

import csv
import os
import time

import torch
import torch.nn as nn

import visualizacion as vis

# BCEWithLogitsLoss = Sigmoide + BCE en una sola operación (más estable
# numéricamente que hacer sigmoide y luego BCELoss por separado).
criterio = nn.BCEWithLogitsLoss()


# =============================================================================
#  FUNCIÓN DE PÉRDIDA DEL DISCRIMINADOR
# =============================================================================
def perdida_discriminador(logits_reales: torch.Tensor, logits_falsos: torch.Tensor,
                          etiqueta_real: float = 0.9):
    """
    L_D = BCE(D(x), 1) + BCE(D(G(z)), 0)
        = -log D(x) - log(1 - D(G(z)))

    - Primer término: castiga a D si NO reconoce los dibujos reales.
    - Segundo término: castiga a D si se deja engañar por los falsos.

    etiqueta_real = 0.9 ("one-sided label smoothing"): en vez de pedirle a D
    que esté 100% seguro de que un real es real, le pedimos 90%. Así D no se
    vuelve sobreconfiado y le deja gradiente útil al generador.
    Pon 1.0 para la versión teórica pura.
    """
    objetivo_real = torch.full_like(logits_reales, etiqueta_real)   # "son reales"
    objetivo_falso = torch.zeros_like(logits_falsos)                # "son falsos"

    perdida_real = criterio(logits_reales, objetivo_real)
    perdida_falsa = criterio(logits_falsos, objetivo_falso)
    return perdida_real + perdida_falsa, perdida_real, perdida_falsa


# =============================================================================
#  FUNCIÓN DE PÉRDIDA DEL GENERADOR
# =============================================================================
def perdida_generador(logits_falsos: torch.Tensor) -> torch.Tensor:
    """
    L_G = BCE(D(G(z)), 1) = -log D(G(z))

    El generador usa etiquetas de "REAL" (1) para SUS imágenes falsas:
    gana (pérdida baja) cuando el discriminador se las cree.
    Ojo: G nunca ve los dibujos reales; solo aprende a través de lo que
    opina D. Por eso D es, literalmente, el "profesor" de G.
    """
    objetivo = torch.ones_like(logits_falsos)
    return criterio(logits_falsos, objetivo)


# =============================================================================
#  BUCLE DE ENTRENAMIENTO
# =============================================================================
def entrenar(generador, discriminador, cargador, args, dispositivo, carpeta):
    """
    Entrena la GAN alternando, en cada lote:
        Paso 1) actualizar D con reales + falsas
        Paso 2) actualizar G para engañar a D
    Guarda una cuadrícula de imágenes por época y el historial de valores.
    """
    # OPTIMIZADORES: Adam con lr=0.0002 y beta1=0.5 (valores del paper DCGAN).
    # Cada red tiene su PROPIO optimizador: solo actualiza sus propios pesos.
    opt_G = torch.optim.Adam(generador.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminador.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Ruido FIJO: los mismos 64 vectores z en todas las épocas. Así las
    # cuadrículas son comparables y se ve cómo EVOLUCIONA el mismo dibujo.
    ruido_fijo = torch.randn(64, args.z, device=dispositivo)

    historial = []          # una fila por época (para CSV y gráficas)
    perdidas_iter_D = []    # una entrada por lote (curva detallada)
    perdidas_iter_G = []
    rutas_epocas = []

    print("\n" + "=" * 92)
    print(f" ENTRENAMIENTO  |  {args.epocas} épocas  |  dispositivo: {dispositivo}")
    print("=" * 92)
    print(" Cómo leerlo:  D(x) = prob. que D da a los REALES (ideal ~0.6-0.8)")
    print("               D(G(z)) = prob. que D da a los FALSOS (ideal ~0.3-0.5; si -> 0, D gana fácil)")
    print("-" * 92)

    for epoca in range(1, args.epocas + 1):
        inicio = time.time()
        suma = {"D": 0.0, "D_real": 0.0, "D_falsa": 0.0, "G": 0.0, "Dx": 0.0, "DGz": 0.0}
        n_lotes = 0

        for (reales,) in cargador:
            reales = reales.to(dispositivo)
            tam = reales.size(0)

            # -----------------------------------------------------------------
            # PASO 1: ENTRENAR AL DISCRIMINADOR
            # -----------------------------------------------------------------
            z = torch.randn(tam, args.z, device=dispositivo)   # ruido aleatorio
            falsas = generador(z)                              # G dibuja

            logits_reales = discriminador(reales)
            # .detach(): corta el gradiente hacia G. En este paso SOLO
            # aprende D; no queremos tocar los pesos del generador.
            logits_falsos = discriminador(falsas.detach())

            loss_D, loss_D_real, loss_D_falsa = perdida_discriminador(
                logits_reales, logits_falsos, args.etiqueta_real)

            opt_D.zero_grad()      # borrar gradientes viejos
            loss_D.backward()      # retropropagación: calcular gradientes
            opt_D.step()           # actualizar pesos de D

            # -----------------------------------------------------------------
            # PASO 2: ENTRENAR AL GENERADOR
            # -----------------------------------------------------------------
            # Volvemos a pasar las falsas por D (ya actualizado), esta vez SIN
            # detach: el gradiente atraviesa D y llega hasta los pesos de G.
            logits_para_G = discriminador(falsas)
            loss_G = perdida_generador(logits_para_G)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()           # solo cambian los pesos de G (opt_G solo tiene los de G)

            # ---- registro de valores --------------------------------------
            with torch.no_grad():
                d_x = torch.sigmoid(logits_reales).mean().item()
                d_g_z = torch.sigmoid(logits_falsos).mean().item()
            suma["D"] += loss_D.item()
            suma["D_real"] += loss_D_real.item()
            suma["D_falsa"] += loss_D_falsa.item()
            suma["G"] += loss_G.item()
            suma["Dx"] += d_x
            suma["DGz"] += d_g_z
            perdidas_iter_D.append(loss_D.item())
            perdidas_iter_G.append(loss_G.item())
            n_lotes += 1

        # ---- resumen de la época -------------------------------------------
        segundos = time.time() - inicio
        fila = {
            "epoca": epoca,
            "perdida_D": suma["D"] / n_lotes,
            "perdida_D_real": suma["D_real"] / n_lotes,
            "perdida_D_falsa": suma["D_falsa"] / n_lotes,
            "perdida_G": suma["G"] / n_lotes,
            "D_x": suma["Dx"] / n_lotes,
            "D_G_z": suma["DGz"] / n_lotes,
            "segundos": segundos,
        }
        historial.append(fila)
        print(f" Época {epoca:3d}/{args.epocas} | Pérdida D: {fila['perdida_D']:.4f} "
              f"(real {fila['perdida_D_real']:.3f} + falsa {fila['perdida_D_falsa']:.3f}) | "
              f"Pérdida G: {fila['perdida_G']:.4f} | D(x): {fila['D_x']:.3f} | "
              f"D(G(z)): {fila['D_G_z']:.3f} | {segundos:.1f}s")

        # ---- cuadrícula de esta época con el ruido fijo --------------------
        with torch.no_grad():
            muestras = generador(ruido_fijo)
        ruta = os.path.join(carpeta, "epocas", f"epoca_{epoca:03d}.png")
        vis.guardar_cuadricula(
            muestras, ruta,
            f"GAN Quick, Draw! — categoría: '{args.categoria}' — época {epoca}/{args.epocas}")
        rutas_epocas.append(ruta)

    # ---- guardar historial en CSV (se abre con Excel) -----------------------
    ruta_csv = os.path.join(carpeta, "historial_entrenamiento.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(historial[0].keys()))
        escritor.writeheader()
        escritor.writerows(historial)

    return historial, perdidas_iter_D, perdidas_iter_G, rutas_epocas
