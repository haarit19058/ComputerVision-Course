import qrcode
from pathlib import Path

text = input("Enter text: ")

# create QR code with no white padding
qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=10,
    border=1,
)

qr.add_data(text)
qr.make(fit=True)

img = qr.make_image()
img.save("./imgs/qrcode.png")

# save the path to .env
qr_path = "./imgs/qrcode.png"
env_file = Path(".env")

lines = env_file.read_text().splitlines() if env_file.exists() else []

for i, line in enumerate(lines):
    if line.startswith("QRCODE_PATH="):
        lines[i] = f"QRCODE_PATH={qr_path}"
        break
else:
    lines.append(f"QRCODE_PATH={qr_path}")

env_file.write_text("\n".join(lines) + "\n")

print("QR code saved as ./imgs/qrcode.png")