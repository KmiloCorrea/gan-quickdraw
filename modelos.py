"""
modelos.py
==========
AQUÍ SE CREAN LAS DOS REDES NEURONALES DE LA GAN.

    - class Generador     -> la red que DIBUJA (crea imágenes falsas a partir de ruido)
    - class Discriminador -> la red que JUZGA  (dice si un dibujo es real o falso)

Arquitectura usada: DCGAN (Deep Convolutional GAN, Radford et al., 2015),
reducida para imágenes de 28x28 en escala de grises (las de Quick, Draw!).

Al final del archivo hay dos funciones de apoyo para la tarea:
    - resumen_modelo()      -> tabla capa por capa con forma de salida y nº de parámetros
    - estadisticas_pesos()  -> valores reales de los pesos (media, desv., mín., máx.)
"""

from collections import OrderedDict

import torch
import torch.nn as nn


# =============================================================================
#  RED NEURONAL GENERADORA
# =============================================================================
class Generador(nn.Module):
    """
    Recibe un vector de ruido aleatorio z (por defecto 100 números ~ N(0,1))
    y lo transforma, paso a paso, en una imagen de 1 x 28 x 28.

    Flujo de formas:
        z (100)
          -> Densa            -> 6272 valores (= 128 canales x 7 x 7)
          -> Reformar         -> 128 x 7 x 7     (un "mapa" pequeño y grueso)
          -> ConvTranspuesta  -> 64 x 14 x 14    (duplica el tamaño)
          -> ConvTranspuesta  -> 1 x 28 x 28     (duplica otra vez = imagen final)
          -> Tanh             -> píxeles en el rango [-1, 1]
    """

    def __init__(self, dim_latente: int = 100):
        super().__init__()
        self.dim_latente = dim_latente

        # nn.Sequential ejecuta las capas en orden. Les ponemos nombre
        # (OrderedDict) para que el resumen de parámetros sea legible.
        self.red = nn.Sequential(OrderedDict([
            # ---- BLOQUE 1: del ruido a un mapa de 128 x 7 x 7 -----------------
            # Capa densa (totalmente conectada): cada uno de los 100 valores de z
            # se conecta con cada una de las 6272 salidas.
            # bias=False porque la BatchNorm de abajo ya trae su propio
            # desplazamiento (beta); tener los dos sería redundante.
            ("capa1_densa", nn.Linear(dim_latente, 128 * 7 * 7, bias=False)),
            # BatchNorm: normaliza las activaciones del lote (media 0, var 1).
            # Estabiliza mucho el entrenamiento de las GAN.
            ("capa1_batchnorm", nn.BatchNorm1d(128 * 7 * 7)),
            # ReLU: deja pasar los positivos y pone a 0 los negativos.
            ("capa1_relu", nn.ReLU()),
            # Reformar: convierte el vector de 6272 en un "cubo" 128 x 7 x 7.
            # No tiene parámetros, solo reorganiza los datos.
            ("capa1_reformar", nn.Unflatten(1, (128, 7, 7))),

            # ---- BLOQUE 2: 128 x 7 x 7  ->  64 x 14 x 14 -----------------------
            # Convolución transpuesta ("deconvolución"): AGRANDA la imagen.
            # kernel 4, stride 2, padding 1  => el alto y ancho se duplican.
            ("capa2_convtranspuesta", nn.ConvTranspose2d(
                in_channels=128, out_channels=64,
                kernel_size=4, stride=2, padding=1, bias=False)),
            ("capa2_batchnorm", nn.BatchNorm2d(64)),
            ("capa2_relu", nn.ReLU()),

            # ---- BLOQUE 3: 64 x 14 x 14  ->  1 x 28 x 28 (IMAGEN FINAL) ---------
            ("capa3_convtranspuesta", nn.ConvTranspose2d(
                in_channels=64, out_channels=1,
                kernel_size=4, stride=2, padding=1)),
            # Tanh: comprime la salida al rango [-1, 1], el mismo rango al que
            # normalizamos los dibujos reales en datos.py.
            ("capa3_tanh", nn.Tanh()),
        ]))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (lote, dim_latente)  ->  imagen: (lote, 1, 28, 28)
        return self.red(z)


# =============================================================================
#  RED NEURONAL DISCRIMINADORA
# =============================================================================
class Discriminador(nn.Module):
    """
    Recibe una imagen de 1 x 28 x 28 (real o generada) y devuelve UN número:
    el "logit" de que la imagen sea REAL.

        logit grande y positivo  -> "creo que es real"
        logit grande y negativo  -> "creo que es falsa"

    Para pasarlo a probabilidad (0 a 1) se aplica sigmoide. No la ponemos
    aquí porque la función de pérdida BCEWithLogitsLoss ya la aplica por
    dentro de forma numéricamente más estable (ver entrenamiento.py).

    Flujo de formas:
        imagen 1 x 28 x 28
          -> Conv  -> 64 x 14 x 14
          -> Conv  -> 128 x 7 x 7
          -> Aplanar -> 6272
          -> Densa -> 1 (logit)
    Es casi el espejo del generador: él agranda, este reduce.
    """

    def __init__(self):
        super().__init__()

        self.red = nn.Sequential(OrderedDict([
            # ---- BLOQUE 1: 1 x 28 x 28  ->  64 x 14 x 14 ------------------------
            # Convolución: 64 filtros de 4x4 que buscan trazos, bordes, curvas.
            # stride 2 => reduce el tamaño a la mitad.
            ("capa1_conv", nn.Conv2d(
                in_channels=1, out_channels=64,
                kernel_size=4, stride=2, padding=1)),
            # LeakyReLU: como ReLU pero deja pasar un 20% de los negativos.
            # Evita "neuronas muertas" y ayuda a que el gradiente llegue
            # hasta el generador.
            ("capa1_leakyrelu", nn.LeakyReLU(0.2)),
            # Dropout: apaga al azar el 30% de las neuronas en cada paso.
            # Frena al discriminador para que no le gane demasiado rápido
            # al generador (si D gana del todo, G deja de aprender).
            ("capa1_dropout", nn.Dropout(0.3)),

            # ---- BLOQUE 2: 64 x 14 x 14  ->  128 x 7 x 7 ------------------------
            ("capa2_conv", nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=4, stride=2, padding=1, bias=False)),
            ("capa2_batchnorm", nn.BatchNorm2d(128)),
            ("capa2_leakyrelu", nn.LeakyReLU(0.2)),
            ("capa2_dropout", nn.Dropout(0.3)),

            # ---- BLOQUE 3: decisión final ---------------------------------------
            ("capa3_aplanar", nn.Flatten()),            # 128x7x7 -> 6272
            ("capa3_densa", nn.Linear(128 * 7 * 7, 1)),  # 6272 -> 1 logit
        ]))

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        # imagen: (lote, 1, 28, 28)  ->  logits: (lote, 1)
        return self.red(imagen)


# =============================================================================
#  INICIALIZACIÓN DE PESOS (recomendación del paper DCGAN)
# =============================================================================
def inicializar_pesos(capa: nn.Module) -> None:
    """
    Se aplica con  red.apply(inicializar_pesos).
    Pesos de convoluciones y densas ~ Normal(media=0, desv=0.02).
    BatchNorm: gamma ~ Normal(1, 0.02) y beta = 0.
    """
    if isinstance(capa, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(capa.weight, mean=0.0, std=0.02)
        if capa.bias is not None:
            nn.init.zeros_(capa.bias)
    elif isinstance(capa, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(capa.weight, mean=1.0, std=0.02)
        nn.init.zeros_(capa.bias)


# =============================================================================
#  HERRAMIENTAS PARA VER LOS PARÁMETROS (lo que pide la tarea)
# =============================================================================
def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    """Devuelve (parámetros totales, parámetros entrenables)."""
    total = sum(p.numel() for p in modelo.parameters())
    entrenables = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    return total, entrenables


def resumen_modelo(modelo: nn.Module, entrada_ejemplo: torch.Tensor, titulo: str) -> str:
    """
    Pasa una entrada de ejemplo por la red y anota, para cada capa:
    nombre, tipo, forma de la salida y cuántos parámetros (pesos + sesgos) tiene.
    Devuelve el texto de la tabla (para imprimirlo y guardarlo en un .txt).
    """
    filas = []
    ganchos = []

    def crear_gancho(nombre_capa):
        # Un "forward hook" es una función que PyTorch llama justo después
        # de que la capa calcula su salida. Lo usamos para espiar la forma.
        def gancho(modulo, entrada, salida):
            n_params = sum(p.numel() for p in modulo.parameters(recurse=False))
            filas.append((nombre_capa, modulo.__class__.__name__,
                          tuple(salida.shape[1:]), n_params))
        return gancho

    for nombre_capa, modulo in modelo.named_modules():
        es_hoja = len(list(modulo.children())) == 0
        if es_hoja:
            nombre_corto = nombre_capa.replace("red.", "")
            ganchos.append(modulo.register_forward_hook(crear_gancho(nombre_corto)))

    estaba_entrenando = modelo.training
    modelo.eval()                      # eval: BatchNorm/Dropout en modo inferencia
    with torch.no_grad():
        modelo(entrada_ejemplo)
    modelo.train(estaba_entrenando)    # lo dejamos como estaba
    for g in ganchos:
        g.remove()

    total, entrenables = contar_parametros(modelo)
    linea = "-" * 84
    texto = [linea, f" {titulo}", linea,
             f" {'Capa':<26}{'Tipo':<20}{'Forma de salida':<24}{'Parámetros':>12}",
             linea,
             f" {'(entrada)':<26}{'':<20}{str(tuple(entrada_ejemplo.shape[1:])):<24}{'':>12}"]
    for nombre, tipo, forma, n in filas:
        texto.append(f" {nombre:<26}{tipo:<20}{str(forma):<24}{n:>12,}")
    texto += [linea,
              f" Parámetros totales:     {total:>12,}",
              f" Parámetros entrenables: {entrenables:>12,}",
              linea]
    return "\n".join(texto)


def estadisticas_pesos(modelo: nn.Module, titulo: str) -> str:
    """
    Muestra los VALORES de cada tensor de parámetros de la red:
    forma, media, desviación estándar, mínimo y máximo.
    Sirve para comparar los pesos antes y después de entrenar.
    """
    linea = "-" * 92
    texto = [linea, f" {titulo}", linea,
             f" {'Parámetro':<34}{'Forma':<20}{'Media':>9}{'Desv.':>9}{'Mín.':>10}{'Máx.':>10}",
             linea]
    for nombre, p in modelo.named_parameters():
        datos = p.detach().float().cpu()
        # (un tensor de 1 solo valor no tiene desviación estándar -> 0)
        desv = datos.std().item() if datos.numel() > 1 else 0.0
        texto.append(
            f" {nombre.replace('red.', ''):<34}{str(tuple(datos.shape)):<20}"
            f"{datos.mean().item():>9.4f}{desv:>9.4f}"
            f"{datos.min().item():>10.4f}{datos.max().item():>10.4f}")
    texto.append(linea)
    return "\n".join(texto)
