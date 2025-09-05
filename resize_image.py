import os
import argparse
from PIL import Image

def resize_and_save(input_folder, output_folder, size, mode="fit"):
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        save_path = os.path.join(output_folder, class_name)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB")
                if mode == "fit":
                    img.thumbnail(size)  # jaga aspect ratio
                    background = Image.new("RGB", size, (255, 255, 255))
                    offset = ((size[0] - img.size[0]) // 2, (size[1] - img.size[1]) // 2)
                    background.paste(img, offset)
                    img = background
                else:
                    img = img.resize(size)  # langsung resize paksa

                img.save(os.path.join(save_path, img_name))
                print(f"[OK] {img_name} -> {save_path}")
            except Exception as e:
                print(f"[ERROR] {img_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Folder input dataset (boleh banyak, misal train val)")
    parser.add_argument("--output", required=True, help="Folder output dataset resized")
    parser.add_argument("--size", type=int, nargs=2, default=[224,224], help="Ukuran resize, default 224x224")
    parser.add_argument("--mode", choices=["fit", "resize"], default="fit", help="Mode resize")
    args = parser.parse_args()

    for folder in args.input:
        folder_name = os.path.basename(folder)
        output_path = os.path.join(args.output, folder_name)
        os.makedirs(output_path, exist_ok=True)
        resize_and_save(folder, output_path, tuple(args.size), args.mode)
