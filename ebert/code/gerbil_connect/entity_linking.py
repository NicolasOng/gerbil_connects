import os

from gerbil_connect.server_template import parse_args, load_model, ebert_model

args = parse_args()
print(os.uname(), flush = True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", None), flush = True)
print(args, flush = True)

model = load_model(args)

output = ebert_model("")

print(output)
