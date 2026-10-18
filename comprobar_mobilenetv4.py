import timm

model_name = "mobilenetv4_conv_medium"

print(f"Comprobando '{model_name}' en timm v{timm.__version__}...\n")

# Verificar si el modelo existe en el catálogo local de timm
available_models = timm.list_models(model_name)
is_available = model_name in available_models
print(f"Modelo disponible en el catálogo: {is_available}")

if is_available:
    print("Descargando pesos preentrenados e instanciando el backbone...")
    # Instanciar el modelo con los pesos de Hugging Face
    model = timm.create_model(model_name, pretrained=True)
    
    print(" Modelo instanciado correctamente.")
    print(f"Tipo de objeto: {type(model)}")
    
    # Calcular parámetros totales para confirmar la carga completa
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {num_params:,}")
else:
    print(" El modelo no se encuentra en esta versión de timm.")