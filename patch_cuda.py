import os

def replace_in_dir(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.endswith('.py'):
                continue
            fpath = os.path.join(root, file)
            with open(fpath, 'r') as f:
                content = f.read()
            if '"cuda"' in content or "'cuda'" in content:
                content = content.replace('"cuda"', '("cuda" if __import__("torch").cuda.is_available() else ("mps" if __import__("torch").backends.mps.is_available() else "cpu"))')
                content = content.replace("'cuda'", '("cuda" if __import__("torch").cuda.is_available() else ("mps" if __import__("torch").backends.mps.is_available() else "cpu"))')
                with open(fpath, 'w') as f:
                    f.write(content)

replace_in_dir('/tmp/sam3/sam3')
print("Patched.")
