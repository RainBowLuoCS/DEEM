import os

os.system("wget https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip .")
os.system("unzip val.zip")
os.system("rm val.zip")

os.system("wget https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip .")
os.system("unzip Annotations.zip")
os.system("rm Annotations.zip")

# import json

# a=json.load(open("./datasets/VizWiz/val.json"))
# print(a)
