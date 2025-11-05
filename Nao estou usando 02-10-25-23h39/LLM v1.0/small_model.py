from tensorflow.keras.models import load_model
m = load_model("modelo_ls14.h5", compile=False)
print("n inputs:", len(m.inputs))
print("input shapes:", [tuple(inp.shape) for inp in m.inputs])
