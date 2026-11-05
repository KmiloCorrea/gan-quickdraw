"""
generar.py
==========
Usa el GENERADOR YA ENTRENADO (no vuelve a entrenar) para crear dibujos nuevos.
Útil para la sustentación: genera al instante frente al profe.

    uv run generar.py                        (usa resultados/cat/generador.pt)
    uv run generar.py --categoria apple
    uv run generar.py --interpolar           (transición suave entre dos dibujos)
"""

import argparse
import os

import torch

import visualizacion as vis
from modelos import Generador


def main():
    p = argparse.ArgumentParser(description="Genera dibujos con un generador ya entrenado")
    p.add_argument("--categoria", default="cat")
    p.add_argument("--cantidad", type=int, default=64)
    p.add_argument("--interpolar", action="store_true",
                   help="Recorre el espacio latente de un vector z a otro")
    args = p.parse_args()

    carpeta = os.path.join("resultados", args.categoria.replace(" ", "_"))
    ruta_modelo = os.path.join(carpeta, "generador.pt")
    if not os.path.exists(ruta_modelo):
        raise SystemExit(f"No existe {ruta_modelo}. Primero entrena: "
                         f"uv run main.py --categoria \"{args.categoria}\"")

    punto = torch.load(ruta_modelo, map_location="cpu")
    generador = Generador(dim_latente=punto["dim_latente"])   # se recrea la red generadora...
    generador.load_state_dict(punto["estado"])                # ...y se le cargan los pesos aprendidos
    # Se deja en modo entrenamiento (BatchNorm con estadísticas del lote), igual
    # que como se generaron las imágenes durante el entrenamiento.

    with torch.no_grad():
        if args.interpolar:
            # Dos puntos aleatorios del espacio latente y 8 pasos intermedios
            # en línea recta entre ellos. Si la GAN aprendió bien, el dibujo
            # cambia de forma suave (no salta al azar).
            filas = []
            for _ in range(8):
                z1, z2 = torch.randn(1, generador.dim_latente), torch.randn(1, generador.dim_latente)
                pasos = torch.linspace(0, 1, 8).unsqueeze(1)
                filas.append((1 - pasos) * z1 + pasos * z2)
            z = torch.cat(filas)
            nombre, titulo = "interpolacion.png", "Interpolación en el espacio latente (cada fila: z1 -> z2)"
        else:
            z = torch.randn(args.cantidad, generador.dim_latente)
            nombre, titulo = "generados_nuevos.png", "Dibujos nuevos"
        imagenes = generador(z)

    ruta = os.path.join(carpeta, nombre)
    vis.guardar_cuadricula(imagenes, ruta, f"{titulo} — categoría: '{punto['categoria']}'")
    print(f"Guardado: {os.path.abspath(ruta)}")


if __name__ == "__main__":
    main()
