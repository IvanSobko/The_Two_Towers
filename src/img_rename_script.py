import os


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


i = 0
for file in files("../img/"):
    if "scene" in file:
        num = str(i)
        num = num.zfill(3)
        os.rename("../img/" + file, f"../img/scene_{num}.jpeg")
        i = i + 1
