import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import random
import math
import torch
import torch.nn as nn
from sklearn.utils import class_weight

# ----------------
# GLOBALS
# ----------------
AUM = 0


def select_patch(datos, sizex, sizey, x, y):
    x1 = int(x) - int(sizex / 2)
    x2 = int(x) + int(math.ceil(sizex / 2))
    y1 = int(y) - int(sizey / 2)
    y2 = int(y) + int(math.ceil(sizey / 2))
    # En caso de que los índices del patch se salgan del imagen, se desplaza para coger lo más cercano al borde posible
    # Puede comentarse si al seleccionar muestras se descomenta la línea que asegura que no se salga
    if x1 < 0:
        x1 = 0
        x2 = sizex
    if x2 > datos.shape[2]:
        x2 = datos.shape[2]
        x1 = datos.shape[2] - sizex
    if y1 < 0:
        y1 = 0
        y2 = sizey
    if y2 > datos.shape[1]:
        y2 = datos.shape[1]
        y1 = datos.shape[1] - sizey

    patch = datos[:, y1:y2, x1:x2]
    return patch


# Esta parte tarda mucho, mejor la preprocesamos en C
def seg_center(seg, H, V):
    print("* Segment centers (tarda mucho)")
    nseg = 0
    for i in range(H * V):
        if seg[i] > nseg:
            nseg = seg[i]
    nseg = nseg + 1
    print("  segments:", nseg)
    xmin = [H * V] * nseg
    xmax = [0] * nseg
    ymin = [H * V] * nseg
    ymax = [0] * nseg
    for i in range(H * V):
        x = i % H
        y = i // H
        s = seg[i]
        if x < xmin[s]:
            xmin[s] = x
        if y < ymin[s]:
            ymin[s] = y
        if x > xmax[s]:
            xmax[s] = x
        if y > ymax[s]:
            ymax[s] = y
    center = np.zeros(nseg, dtype=np.uint32)
    for s in range(nseg):
        y = (ymin[s] + ymax[s]) // 2
        x = (xmin[s] + xmax[s]) // 2
        center[s] = y * H + x
    return (center, nseg)


def select_training_samples_seg(truth, center, H, V, sizex, sizey, porcentaje):
    print("* Select training samples")
    # hacemos una lista con las clases, pero puede haber clases vacias
    nclases = 0
    nclases_no_vacias = 0
    N = len(truth)
    for i in truth:
        if i > nclases:
            nclases = i
    print("  nclasses:", nclases)
    lista = [0] * nclases
    for i in range(nclases):
        lista[i] = []
    xmin = int(sizex / 2)
    xmax = H - int(math.ceil(sizex / 2))
    ymin = int(sizey / 2)
    ymax = V - int(math.ceil(sizey / 2))
    for ind in center:
        i = ind // H
        j = ind % H
        # ¿Y si el segmento es más pequeño que un patch?
        # Si se comenta esta linea, los patch deben rellenarse con lo que falte
        # if(i<ymin or i>ymax or j<xmin or j>xmax): continue
        if truth[ind] > 0:
            lista[truth[ind] - 1].append(ind)
    for i in range(nclases):
        random.shuffle(lista[i])
    # seleccionamos muestras para train, validacion y test
    print("  Class  # :   total | train |   val |    test")
    train = []
    val = []
    test = []
    for i in range(nclases):
        # tot0: numero muestras entrenamiento, tot1: validacion
        if porcentaje[0] >= 1:
            tot0 = porcentaje[0]
        else:
            tot0 = int(porcentaje[0] * len(lista[i]))
        if tot0 >= len(lista[i]):
            tot0 = len(lista[i]) // 2
        if tot0 < 0 and len(lista[i]) > 0:
            tot0 = 1
        if tot0 != 0:
            nclases_no_vacias += 1
        if porcentaje[1] >= 1:
            tot1 = porcentaje[1]
        else:
            tot1 = int(porcentaje[1] * len(lista[i]))
        if tot1 >= len(lista[i]):
            tot1 = len(lista[i]) // 2
        if tot1 < 1 and len(lista[i]) > 0:
            tot1 = 0
        for j in range(len(lista[i])):
            if j < tot0:
                train.append(lista[i][j])
            elif j < tot0 + tot1:
                val.append(lista[i][j])
            # testeamos los segmentos que no esten en el entrenamiento
            # else: test.append(lista[i][j])
            # testeamos todos los segmentos
            test.append(lista[i][j])
        print(
            "  Class",
            f"{i+1:2d}",
            ":",
            f"{len(lista[i]):7d}",
            "|",
            f"{tot0:5d}",
            "|",
            f"{tot1:5d}",
            "|",
            f"{len(lista[i])-tot0-tot1:7d}",
        )
    return (train, val, test, nclases, nclases_no_vacias)


# -----------------------------------------------------------------
# PYTORCH - SETS
# -----------------------------------------------------------------


# cogemos muestras sin ground-truth (dadas por el indice samples)
class HyperAllDataset(Dataset):
    def __init__(self, datos, samples, H, V, sizex, sizey):
        self.datos = datos
        self.samples = samples
        self.H = H
        self.V = V
        self.sizex = sizex
        self.sizey = sizey
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees=30)
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        datos = self.datos
        H = self.H
        V = self.V
        sizex = self.sizex
        sizey = self.sizey
        x = self.samples[idx] % H
        y = int(self.samples[idx] / H)
        patch = select_patch(datos, sizex, sizey, x, y)
        if AUM == 1:
            patch = self.transform(patch)
        return patch


# ----------------


# cogemos muestras con ground-truth (dadas por el indice samples)
class HyperDataset(Dataset):
    def __init__(self, datos, truth, samples, H, V, sizex, sizey):
        self.datos = datos
        self.truth = truth
        self.samples = samples
        self.H = H
        self.V = V
        self.sizex = sizex
        self.sizey = sizey
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(degrees=30)
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        datos = self.datos
        truth = self.truth
        H = self.H
        V = self.V
        sizex = self.sizex
        sizey = self.sizey
        x = self.samples[idx] % H
        y = int(self.samples[idx] / H)
        patch = select_patch(datos, sizex, sizey, x, y)
        if AUM == 1:
            patch = self.transform(patch)
        # renumeramos porque la red clasifica tambien la clase 0
        return (patch, truth[self.samples[idx]] - 1)


# -----------------------------------------------------------------
# PYTORCH - UTIL
# -----------------------------------------------------------------


# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear
def signal_handler(sig, frame):
    print("\n* Ctrl+C. Exit training")
    global endTrain
    endTrain = True


# For updating learning rate manual
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# calcula los promedios de precisiones
def accuracy_mean_deviation(OA, AA, aa):
    n = len(OA)
    nclases = len(aa[0])
    print("* Means and deviations (%d exp):" % (n))
    if n == 0:
        return (0, 0, 0, 0, [0] * nclases, [0] * nclases)
    elif n == 1:
        return (OA[0], AA[0], 0, 0, aa[0], [0] * nclases)
    else:
        # medias
        OAm = 0
        AAm = 0
        aam = [0] * nclases
        for i in range(n):
            OAm += OA[i]
            AAm += AA[i]
            for j in range(1, nclases):
                aam[j] += aa[i][j]
        OAm /= n
        AAm /= n
        for j in range(1, nclases):
            aam[j] /= n
        # desviaciones, usamos la formula que divide entre (n-1)
        OAd = 0
        AAd = 0
        aad = [0] * nclases
        for i in range(n):
            OAd += (OA[i] - OAm) * (OA[i] - OAm)
            AAd += (AA[i] - AAm) * (AA[i] - OAm)
            for j in range(1, nclases):
                aad[j] += (aa[i][j] - aam[j]) * (aa[i][j] - aam[j])
        OAd = math.sqrt(OAd / (n - 1))
        AAd = math.sqrt(AAd / (n - 1))
        for j in range(1, nclases):
            aad[j] = math.sqrt(aad[j] / (n - 1))
        for j in range(1, nclases):
            print("  Class %02d: %02.02f+%02.02f" % (j, aam[j], aad[j]))
        print("  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f" % (OAm, OAd, AAm, AAd))
        # devolvemos las medias y desviaciones
        return (OAm, AAm, OAd, AAd, aam, aad)
