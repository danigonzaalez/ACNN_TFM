import os
import random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from utils.dataloader import (
    read_raw,
    read_pgm,
    read_seg,
    read_seg_centers,
    save_pgm,
    generate_rgb_map,
)
from utils.torch_utils import (
    seg_center,
    select_training_samples_seg,
    accuracy_mean_deviation,
    update_lr,
    HyperDataset,
)
from utils.general_utils import ad_accuracies

from torchviz import make_dot

EXP = 5  # numero de experimentos
EPOCHS = 100  # EPOCHS de entrenamiente del clasificador, default=100
SAMPLES = [
    0.15,
    0.05,
]  # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05)
BATCH = 100  # batch_size, defecto 100
ADA = 3  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM = 1  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET = 0  # experimentos: 0-aleatorios, 1-deterministas
TEST = 1  # 0-validacion, 1-test
SIZEX = 32  # tamano del patch (defecto=32)  **[NO CAMBIES ESTO]**


COMPUTE_SEG_CENTERS = 0
BASEPATH = "../data/z1/"

OUTPUTDIR = "./results/"
DATASET = BASEPATH + "z1.raw"
GT = BASEPATH + "z1_gt_reetiquetado.pgm"
SEG = BASEPATH + "seg.raw"
CENTER = BASEPATH + "seg_centers.raw"


class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class CNN21(nn.Module):
    def __init__(self, N1, N2, N3, N4, N5, N6, N7, D1, D2, D3):
        super(CNN21, self).__init__()

        # First convolution layer with attention
        attention_dim = 2048
        self.conv1 = nn.Conv2d(N1, attention_dim, kernel_size=3, stride=1, padding=1)
        self.attention1 = Attention(attention_dim)

        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(attention_dim, N2, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(N2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=D1),
        )
        self.attention2 = Attention(N2)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(N2, N3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(N3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=D2),
        )
        self.attention3 = Attention(N3)

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(N3, N4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(N4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=D3),
        )
        self.attention4 = Attention(N4)

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(N4, N5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(N5),
            nn.ReLU(),
        )
        self.attention5 = Attention(N5)

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv2d(N5, N6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(N6),
            nn.ReLU(),
        )
        self.attention6 = Attention(N6)

        # Fully connected layers
        self.fc1 = nn.Linear(
            N6 * 4 * 4, N7
        )  # Adjust dimensions based on the output of the convolutional layers
        self.fc2 = nn.Linear(N7, 2)  # Output layer (assuming 10 classes)

    def forward(self, x):
        # Apply first convolution and attention
        x = self.conv1(x)
        x = self.attention1(x)

        # Apply layer1 and attention
        x = self.layer1(x)
        x = self.attention2(x)

        # Apply layer2 and attention
        x = self.layer2(x)
        x = self.attention3(x)

        # Apply layer3 and attention
        x = self.layer3(x)
        x = self.attention4(x)

        # Apply layer4 and attention
        x = self.layer4(x)
        x = self.attention5(x)

        # Apply layer5 and attention
        x = self.layer5(x)
        x = self.attention6(x)

        # Flatten and apply fully connected layers
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


# -----------------------------------------------------------------
# PYTORCH - MAIN
# -----------------------------------------------------------------


def main(exp, DATASET=DATASET, GT=GT, SEG=SEG, CENTER=CENTER, config=None):

    if config is not None:
        print("* Using config file")

        # print loaded configuration as dictionary
        # print('Loaded configuration:')
        # for key, value in config.items():
        #   print(f'{key}: {value}')

        global EPOCHS, BATCH, SIZEX, DET, AUM, ViTsize, ViTtype, TEST, OUTPUTDIR, ADA, COMPUTE_SEG_CENTERS, SAMPLES
        exp = config.get("EXP_NUMBER", exp)
        EPOCHS = config.get("EPOCHS", EPOCHS)
        SAMPLES = config.get("SAMPLES", SAMPLES)
        BATCH = config.get("BATCH", BATCH)
        SIZEX = config.get("SIZEX", SIZEX)
        DET = config.get("DET", DET)
        AUM = config.get("AUM", AUM)
        ADA = config.get("ADA", ADA)
        TEST = config.get("TEST", TEST)
        OUTPUTDIR = config.get("OUTPUTDIR", OUTPUTDIR)
        os.makedirs(OUTPUTDIR, exist_ok=True)
        COMPUTE_SEG_CENTERS = config.get("COMPUTE_SEG_CENTERS", COMPUTE_SEG_CENTERS)

    print("* CNN21 exp: " + str(exp))
    time_start = time.time()
    # 1. Device configuration
    cuda = True if torch.cuda.is_available() else False
    print("* cuda:", cuda)
    device = torch.device("cuda" if cuda else "cpu")
    if torch.backends.cudnn.is_available():
        print("* Activando CUDNN")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.beBhmark = True
    # experimentos deterministas o aleatorios
    if DET == 1:
        SEED = 0
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        if cuda == False:
            torch.use_deterministic_algorithms(True)
            g = torch.Generator()
            g.manual_seed(SEED)
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 2. Load datos
    datos, H, V, B = read_raw(DATASET)
    truth, H1, V1 = read_pgm(GT)
    seg, H2, V2 = read_seg(SEG)
    # necesitamos los datos en band-vector para hacer convoluciones
    datos = np.transpose(datos, (2, 0, 1))
    # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
    sizex = SIZEX
    sizey = SIZEX

    # 3. Selection training,testing sets
    if COMPUTE_SEG_CENTERS == 1:
        print("* Calculando centros de segmentos")
        center, nseg = seg_center(seg, H, V)  # lento, mejor lo cargamos hecho
    else:
        print("* Cargando centros de segmentos")
        center, H3, V3, nseg = read_seg_centers(CENTER)
    train, val, test, nclases, nclases_no_vacias = select_training_samples_seg(
        truth, center, H, V, sizex, sizey, SAMPLES
    )
    dataset_train = HyperDataset(datos, truth, train, H, V, sizex, sizey)
    print("  - train dataset:", len(dataset_train))
    dataset_test = HyperDataset(datos, truth, test, H, V, sizex, sizey)
    print("  - test dataset:", len(dataset_test))
    # Dataloader
    batch_size = BATCH  # defecto 100
    train_loader = DataLoader(dataset_train, batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size, shuffle=False)
    # Si queremos validacion
    if len(val) > 0:
        dataset_val = HyperDataset(datos, truth, val, H, V, sizex, sizey)
        print("  - val dataset:", len(dataset_val))
        val_loader = DataLoader(dataset_val, batch_size, shuffle=False)

    # 4. Hyper parameters
    if ADA == 0:
        lr = 0.001
    else:
        lr = 0.001

    N1 = B  # Dimensión de entrada
    D1 = 2  # Decimación inicial
    H1 = sizex  # Tamaño de los patches de entrada
    N2 = 16  # Dimensión de salida de conv1
    H2 = int(H1 / D1)  # Lado patches después de conv1

    N3 = 32  # Dimensión de salida de conv2
    D2 = 2  # Decimación en conv2
    H3 = int(H2 / D2)  # Lado patches después de conv2

    N4 = 64  # Dimensión de salida de conv3
    D3 = 2  # Decimación en conv3
    H4 = int(H3 / D3)  # Lado patches después de conv3

    N5 = 128  # Dimensión de salida de conv4
    H5 = H4  # No hay decimación en conv4

    N6 = 256  # Dimensión de salida de conv5
    H6 = H5  # No hay decimación en conv5

    N7 = H6 * H6 * N6  # Dimensión de entrada para el FC1
    N8 = 512  # Dimensión del primer fully connected layer
    N9 = nclases  # Dimensión de salida final

    model = CNN21(N1, N2, N3, N4, N5, N6, N8, D1, D2, D3).to(device)

    # 6. Loss, optimizer, and scheduler
    # 6.1 mean-squared error loss
    criterion = nn.CrossEntropyLoss()

    # 6.2 create an optimizer object: Adam optimizer with learning rate lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6.3 scheduler (no es estrictamente necesario)
    if ADA == 2:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[EPOCHS // 2, (5 * EPOCHS) // 6], gamma=0.1
        )
    elif ADA == 3:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=0, verbose=True
        )
    elif ADA == 4:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.99, verbose=True
        )
    else:
        pass

    #     # Count total parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")

    # # Count only trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {trainable_params}")

    # 7. Train the model
    print("* Train CNN21, exp.%d" % (exp))
    global endTrain
    endTrain = False
    # signal.signal(signal.SIGINT,signal_handler)
    total_step = len(train_loader)
    for epoch in tqdm.trange(EPOCHS, desc="epochs"):
        # model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # 7.1. Cogemos muestras para entrenar
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 7.2. Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 7.3. Backward and optimize
            # 7.3.1. reset the gradients (PyTorch accumulates gradients on subsequent backward passes)
            optimizer.zero_grad()
            # 7.3.2. compute accumulated gradients
            loss.backward()
            # 7.3.3. perform parameter update based on current gradients
            optimizer.step()

        # si tenemos validacion usamos estas muestras, si no el propio train
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            if len(val) > 0:
                model.eval()
                losses = []
                acces = []
                for i, (inputs, labels) in enumerate(val_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    acc = torch.mean((outputs.argmax(dim=-1) == labels).float())
                    losses.append(loss.item())
                    acces.append(acc.item())
                print(
                    "  Epoch: %3d/%d, Val. Loss: %.4f, Acc: %.4f"
                    % (
                        epoch,
                        EPOCHS,
                        sum(losses) / len(losses),
                        sum(acces) / len(acces),
                    )
                )
            else:
                print(
                    "  Epoch: %3d/%d, Train Loss: %.4f" % (epoch, EPOCHS, loss.item())
                )

        # Decay learning rate (lo decrementamos cconforme aumentan las iteraciones)
        if ADA == 1 and (epoch + 1) % 20 == 0:
            lr /= 2
            update_lr(optimizer, lr)
        elif ADA > 1:
            scheduler.step()
        if endTrain:
            break

    if TEST == 0:
        return sum(acces) / len(acces)

    # 8. Test the model
    print("* Test CNN21, exp.%d" % (exp))
    output = np.zeros(H * V, dtype=np.uint8)  # mapa de salida de pixels
    # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predicted_cpu = predicted.cpu()
            for i in range(len(predicted_cpu)):
                # queremos que las clases comiencen en 1 en vez de 0
                output[test[total + i]] = np.uint8(predicted_cpu[i] + 1)
            total += labels.size(0)
            if total % 2000 == 0:
                print("  Test: %6d/%d" % (total, len(dataset_test)))
    print("* Generating classif.map")
    for i in range(H * V):
        output[i] = output[center[seg[i]]]
    # eliminamos los centros usados en el entrenamiento
    for i in train:
        output[i] = 0
    for i in val:
        output[i] = 0

    # 9. precisiones por segmentos (excluyendo los usados en el entrenamiento)
    correct = 0
    total = 0
    for i in range(len(center)):
        if output[center[i]] == 0:
            continue
        total += 1
        if output[center[i]] == truth[center[i]]:
            correct = correct + 1
    acc = 100 * correct / total
    print("* Accuracy (segments): %.02f" % (acc))

    # 10. precisiones a nivel de pixel
    correct = 0
    total = 0
    AA = 0
    OA = 0
    class_correct = [0] * (nclases + 1)
    class_total = [0] * (nclases + 1)
    class_aa = [0] * (nclases + 1)
    for i in range(len(output)):
        if output[i] == 0 or truth[i] == 0:
            continue
        total += 1
        class_total[truth[i]] += 1
        if output[i] == truth[i]:
            correct += 1
            class_correct[truth[i]] += 1
    for i in range(1, nclases + 1):
        if class_total[i] != 0:
            class_aa[i] = 100 * class_correct[i] / class_total[i]
        else:
            class_aa[i] = 0
        AA += class_aa[i]
    OA = 100 * correct / total
    AA = AA / nclases_no_vacias
    print("* Accuracy (pixels) exp.%d:" % (exp))
    for i in range(1, nclases + 1):
        print("  Class %02d: %02.02f" % (i, class_aa[i]))
    print("* Accuracy (pixels) exp.%d, OA=%02.02f, AA=%02.02f" % (exp, OA, AA))
    print("  total:", total, "correct:", correct)

    if nclases == 2:
        ad_results_dict = ad_accuracies(truth, output)

    # guardamos la salida
    save_pgm(
        output, H, V, nclases, OUTPUTDIR + "output_cnn21_attention-" + str(exp) + ".pgm"
    )
    # generamos el mapa de clasificacion RGB
    img = generate_rgb_map(output, truth, H, V)
    img.save(OUTPUTDIR + "output_rgb_cnn21_attention-" + str(exp) + ".png")
    # Save the model checkpoint
    torch.save(model.state_dict(), "/tmp/model_cnn21_attention-" + str(exp) + ".ckpt")

    time_end = time.time()
    print("* + time: %.0f s" % (time_end - time_start))
    print("  lr:", lr, "BATCH:", batch_size)

    # Graph visualization of model
    # model.eval()
    # inputs = torch.randn(1, B, sizex, sizey).to(device)
    # outputs = model(inputs)

    # make_dot(outputs.mean(), params=dict(model.named_parameters())).render(
    #     "acnn_torchviz", format="png"
    # )

    return (OA, AA, class_aa, ad_results_dict)


if __name__ == "__main__":
    if TEST == 0:  # validacion (se podria hacer una validacion cruzada de varias vias)
        acces = 0
        for exp in range(EXP):
            acces = acces + main(exp)
        print(
            "* CNN21 SEG EXP:",
            EXP,
            "EPOCHS:",
            EPOCHS,
            "SAMPLES:",
            SAMPLES,
            "ADA:",
            ADA,
            "AUM:",
            AUM,
        )
        print("  VAL: %02.02f" % (100 * acces / EXP))
    else:  # test de 5 experimentos
        OA = [0] * EXP
        AA = [0] * EXP
        aa = [0] * EXP
        for exp in range(EXP):
            OA[exp], AA[exp], aa[exp] = main(exp)
        if EXP > 1:
            accuracy_mean_deviation(OA, AA, aa)
        print(
            "* CNN21 SEG EXP:",
            EXP,
            "EPOCHS:",
            EPOCHS,
            "SAMPLES:",
            SAMPLES,
            "ADA:",
            ADA,
            "AUM:",
            AUM,
        )
