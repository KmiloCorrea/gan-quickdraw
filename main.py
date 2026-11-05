"""
main.py — GAN con Quick, Draw!
==============================
Programa principal. Ejecútalo con:

    uv run main.py                         (categoría por defecto: cat)
    uv run main.py --categoria apple       (otra categoría)
    uv run main.py --epocas 5 --imagenes 10000   (prueba rápida)

Orden de lo que hace:
    1. Descarga y prepara los dibujos de la categoría elegida      (datos.py)
    2. CREA LA RED GENERADORA y LA RED DISCRIMINADORA              (modelos.py)
    3. Muestra sus parámetros (capa por capa) y valores iniciales
    4. Entrena la GAN con sus funciones de pérdida                 (entrenamiento.py)
    5. Guarda resultados: imágenes, gráficas, CSV, pesos           (visualizacion.py)

Todo queda en:  resultados/<categoria>/
"""

import argparse
import os
import time

import torch

import visualizacion as vis
from datos import cargar_quickdraw
from entrenamiento import entrenar
from modelos import (Discriminador, Generador, estadisticas_pesos,
                     inicializar_pesos, resumen_modelo)


def leer_argumentos():
    p = argparse.ArgumentParser(description="GAN (DCGAN) entrenada con dibujos de Quick, Draw!")
    p.add_argument("--categoria", default="cat",
                   help="Categoría de Quick, Draw! en inglés (cat, apple, star, 'smiley face'...)")
    p.add_argument("--epocas", type=int, default=20,
                   help="Pasadas completas por los datos (default 20)")
    p.add_argument("--imagenes", type=int, default=30000,
                   help="Cuántos dibujos usar de la categoría (default 30000)")
    p.add_argument("--lote", type=int, default=128,
                   help="Tamaño de lote / batch size (default 128)")
    p.add_argument("--lr", type=float, default=0.0002,
                   help="Tasa de aprendizaje de Adam para G y D (default 0.0002)")
    p.add_argument("--z", type=int, default=100,
                   help="Dimensión del vector de ruido (espacio latente) (default 100)")
    p.add_argument("--etiqueta-real", type=float, default=0.9, dest="etiqueta_real",
                   help="Etiqueta para los reales en la pérdida de D (0.9 = label smoothing; 1.0 = teórico)")
    p.add_argument("--semilla", type=int, default=42,
                   help="Semilla aleatoria para resultados reproducibles")
    return p.parse_args()


def main():
    args = leer_argumentos()
    torch.manual_seed(args.semilla)

    # Si hay GPU NVIDIA con CUDA disponible la usa; si no, CPU (también funciona).
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    carpeta = os.path.join("resultados", args.categoria.replace(" ", "_"))
    os.makedirs(carpeta, exist_ok=True)

    print("=" * 92)
    print(f" GAN con Quick, Draw!  |  CATEGORÍA USADA: '{args.categoria}'  |  dispositivo: {dispositivo}")
    print("=" * 92)

    # -------------------------------------------------------------------------
    # 1. DATOS
    # -------------------------------------------------------------------------
    cargador, imagenes_reales, total = cargar_quickdraw(
        args.categoria, args.imagenes, args.lote, args.semilla)
    vis.guardar_cuadricula(
        imagenes_reales[:64], os.path.join(carpeta, "00_dibujos_reales.png"),
        f"Dibujos REALES de Quick, Draw! — categoría: '{args.categoria}'")

    # -------------------------------------------------------------------------
    # 2. CREACIÓN DE LAS REDES NEURONALES
    # -------------------------------------------------------------------------
    # >>> AQUÍ SE CREA LA RED NEURONAL GENERADORA (clase en modelos.py) <<<
    generador = Generador(dim_latente=args.z).to(dispositivo)

    # >>> AQUÍ SE CREA LA RED NEURONAL DISCRIMINADORA (clase en modelos.py) <<<
    discriminador = Discriminador().to(dispositivo)

    # Pesos iniciales aleatorios pequeños ~ N(0, 0.02), como en el paper DCGAN
    generador.apply(inicializar_pesos)
    discriminador.apply(inicializar_pesos)

    # -------------------------------------------------------------------------
    # 3. PARÁMETROS DE CADA RED
    # -------------------------------------------------------------------------
    texto_G = resumen_modelo(generador, torch.randn(2, args.z, device=dispositivo),
                             "RED NEURONAL GENERADORA  (ruido z -> dibujo 28x28)")
    texto_D = resumen_modelo(discriminador, torch.randn(2, 1, 28, 28, device=dispositivo),
                             "RED NEURONAL DISCRIMINADORA  (dibujo 28x28 -> ¿real o falso?)")
    texto_hiper = "\n".join([
        "-" * 84,
        " HIPERPARÁMETROS DEL ENTRENAMIENTO",
        "-" * 84,
        f" Categoría Quick, Draw!     : {args.categoria}  ({total:,} dibujos disponibles)",
        f" Dibujos usados             : {min(args.imagenes, total):,}",
        f" Épocas                     : {args.epocas}",
        f" Tamaño de lote             : {args.lote}",
        f" Dimensión del ruido z      : {args.z}",
        f" Optimizador (G y D)        : Adam, lr={args.lr}, betas=(0.5, 0.999)",
        " Función de pérdida         : Entropía cruzada binaria (BCEWithLogitsLoss)",
        f" Etiqueta 'real' para D     : {args.etiqueta_real}",
        f" Dispositivo                : {dispositivo}",
        "-" * 84,
    ])
    texto_pesos_ini_G = estadisticas_pesos(generador, "VALORES INICIALES DE LOS PESOS — GENERADOR")
    texto_pesos_ini_D = estadisticas_pesos(discriminador, "VALORES INICIALES DE LOS PESOS — DISCRIMINADOR")

    for t in (texto_G, texto_D, texto_hiper, texto_pesos_ini_G, texto_pesos_ini_D):
        print("\n" + t)

    with open(os.path.join(carpeta, "01_arquitectura_y_parametros.txt"), "w", encoding="utf-8") as f:
        f.write("\n\n".join([texto_G, texto_D, texto_hiper, texto_pesos_ini_G, texto_pesos_ini_D]))

    # -------------------------------------------------------------------------
    # 4. ENTRENAMIENTO
    # -------------------------------------------------------------------------
    inicio = time.time()
    historial, perd_D, perd_G, rutas_epocas = entrenar(
        generador, discriminador, cargador, args, dispositivo, carpeta)
    minutos = (time.time() - inicio) / 60

    # -------------------------------------------------------------------------
    # 5. RESULTADOS
    # -------------------------------------------------------------------------
    print("\n[resultados] Guardando gráficas e imágenes...")
    vis.graficar_perdidas(perd_D, perd_G, historial,
                          os.path.join(carpeta, "02_funciones_de_perdida.png"), args.categoria)
    vis.graficar_probabilidades(historial,
                                os.path.join(carpeta, "03_salidas_discriminador.png"), args.categoria)

    # Resultado final: 64 dibujos NUEVOS con ruido que la red nunca había visto
    with torch.no_grad():
        finales = generador(torch.randn(64, args.z, device=dispositivo))
    vis.guardar_cuadricula(
        finales, os.path.join(carpeta, "04_resultado_final.png"),
        f"Dibujos GENERADOS por la GAN — categoría: '{args.categoria}' — {args.epocas} épocas")
    vis.crear_gif(rutas_epocas, os.path.join(carpeta, "05_evolucion.gif"))

    vis.histogramas_pesos(generador, os.path.join(carpeta, "06_pesos_generador.png"),
                          "Distribución de los pesos del GENERADOR tras entrenar")
    vis.histogramas_pesos(discriminador, os.path.join(carpeta, "07_pesos_discriminador.png"),
                          "Distribución de los pesos del DISCRIMINADOR tras entrenar")

    texto_pesos_fin_G = estadisticas_pesos(generador, "VALORES FINALES DE LOS PESOS — GENERADOR")
    texto_pesos_fin_D = estadisticas_pesos(discriminador, "VALORES FINALES DE LOS PESOS — DISCRIMINADOR")
    print("\n" + texto_pesos_fin_G + "\n\n" + texto_pesos_fin_D)
    with open(os.path.join(carpeta, "08_pesos_finales.txt"), "w", encoding="utf-8") as f:
        f.write(texto_pesos_fin_G + "\n\n" + texto_pesos_fin_D)

    # Guardar los modelos entrenados (para generar más dibujos con generar.py)
    torch.save({"estado": generador.state_dict(), "dim_latente": args.z,
                "categoria": args.categoria}, os.path.join(carpeta, "generador.pt"))
    torch.save({"estado": discriminador.state_dict(), "categoria": args.categoria},
               os.path.join(carpeta, "discriminador.pt"))

    ultima = historial[-1]
    print("\n" + "=" * 92)
    print(f" LISTO en {minutos:.1f} min  |  categoría '{args.categoria}'  |  "
          f"L_D final {ultima['perdida_D']:.3f}  |  L_G final {ultima['perdida_G']:.3f}  |  "
          f"D(x) {ultima['D_x']:.2f}  |  D(G(z)) {ultima['D_G_z']:.2f}")
    print(f" Todo quedó en la carpeta:  {os.path.abspath(carpeta)}")
    print("=" * 92)


# En Windows este "if" es obligatorio para que PyTorch no relance el script
if __name__ == "__main__":
    main()
