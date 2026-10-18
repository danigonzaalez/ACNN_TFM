import numpy as np
import torch
import struct
from sklearn import preprocessing
from PIL import Image


def read_raw(fichero):
    B, H, V = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=B * H * V, offset=3 * 4, dtype=np.int32)
    print("* Read dataset:", fichero)
    print("  B:", B, "H:", H, "V:", V)
    print("  Read:", len(datos))
    # esta red no necesita realmente normalizar
    datos = preprocessing.minmax_scale(datos)
    print("  min:", datos.min(), "max:", datos.max())
    datos = datos.reshape(V, H, B)
    datos = torch.FloatTensor(datos)
    return (datos, H, V, B)


def read_seg(fichero):
    H, V = np.fromfile(fichero, count=2, dtype=np.uint32)
    datos = np.fromfile(fichero, count=H * V, offset=2 * 4, dtype=np.uint32)
    print("* Read segmentation:", fichero)
    print("  H:", H, "V:", V)
    print("  Read:", len(datos))
    return (datos, H, V)


def read_seg_centers(fichero):
    H, V, nseg = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=H * V, offset=3 * 4, dtype=np.uint32)
    print("* Read centers:", fichero)
    print("  H:", H, "V:", V, "nseg", nseg)
    print("  Read:", len(datos))
    return (datos, H, V, nseg)


def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print("No puedo abrir ", filename)
        exit(0)
    else:
        f.write(struct.pack("i", B))
        f.write(struct.pack("i", H))
        f.write(struct.pack("i", V))
        output = output.reshape(H * V * B)
        for i in range(H * V * B):
            f.write(struct.pack("i", np.int(output[i])))
        f.close()
        print("* Saved file:", filename)


def read_pgm(fichero):
    try:
        pgmf = open(fichero, "rb")
    except IOError:
        print("No puedo abrir ", fichero)
    else:
        assert pgmf.readline().decode() == "P5\n"
        line = pgmf.readline().decode()
        while line[0] == "#":
            line = pgmf.readline().decode()
        H, V = line.split()
        H = int(H)
        V = int(V)
        depth = int(pgmf.readline().decode())
        assert depth <= 255
        raster = []
        for i in range(H * V):
            raster.append(ord(pgmf.read(1)))
        print("* Read GT:", fichero)
        print("  H:", H, "V:", V, "depth:", depth)
        print("  Read:", len(raster))
        return (raster, H, V)


def save_pgm(output, H, V, nclases, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print("No puedo abrir ", filename)
        exit(0)
    else:
        # f.write(b'P5\n')
        cadena = "P5\n" + str(H) + " " + str(V) + "\n" + str(nclases) + "\n"
        f.write(bytes(cadena, "utf-8"))
        f.write(output)
        f.close()
        print("* Saved file:", filename)


def generate_rgb_map(output, truth, H, V):

    # generar mapa de clasificacion RGB
    # Black = true negatives, Red = false negatives, Green = true positives, Blue = false positives
    # print(np.unique(truth))
    # print(np.unique(output))
    output_rgb = np.zeros((V, H, 3), dtype=np.uint8)
    for idx in range(H * V):
        i, j = divmod(idx, H)
        if truth[idx] == 0 or output[idx] == 0:
            output_rgb[i, j] = [0, 0, 0]  # Unlabeled
        elif truth[idx] == 1 and output[idx] == 1:
            output_rgb[i, j] = [0, 0, 0]  # True negatives
        elif truth[idx] == 1 and output[idx] == 2:
            output_rgb[i, j] = [0, 0, 255]  # False positives
        elif truth[idx] == 2 and output[idx] == 1:
            output_rgb[i, j] = [255, 0, 0]  # False negatives
        elif truth[idx] == 2 and output[idx] == 2:
            output_rgb[i, j] = [0, 255, 0]  # True positives

    # save as color image in png format

    img = Image.fromarray(output_rgb, "RGB")

    return img
