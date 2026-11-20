import inspect
from terratorch.models.decoders import UperNetDecoder

print("Firma de UperNetDecoder.__init__:")
print(inspect.signature(UperNetDecoder.__init__))
print("\nDocstring:")
print(UperNetDecoder.__doc__)