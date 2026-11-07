from terratorch.registry import BACKBONE_REGISTRY

model_name = "timm_prithvi_eo_v2_300"

print("Modelo disponible:", model_name in BACKBONE_REGISTRY)

model = BACKBONE_REGISTRY.build(
    model_name,
    pretrained=True
)

print("Modelo instanciado:", type(model))