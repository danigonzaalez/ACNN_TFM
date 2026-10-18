# ACNN

## 1. Crear el entorno virtual

Desde el directorio raíz del proyecto, crea un entorno virtual con Python 3:

```bash
python3 -m venv .venv
```

## 2. Activar el entorno virtual

En Linux o macOS:

```bash
source .venv/bin/activate
```

Para salir del entorno cuando termines:

```bash
deactivate
```

## 3. Instalar los requisitos

Con el entorno virtual activado, instala las dependencias del proyecto:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 4. Ejecutar `main.py`

Desde el directorio raíz del proyecto, ejecuta el programa con el archivo de configuración de ejemplo:

```bash
python main.py --config config.json
```

El programa guarda los registros en `log/` y los resultados en `results/`.
