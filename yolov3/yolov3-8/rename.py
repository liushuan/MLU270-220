import glob
import shutil
img_path = r'/opt/cambricon/yolov3-8/quan_img'

file_lists_jpg = glob.glob(img_path+r"/**/*.jpg",recursive=True)
print("filesize:", len(file_lists_jpg))

for i, file in enumerate(file_lists_jpg):
    file_name = file.split("/")[-1]
    
    new_name = str(i)+".jpg"

    new_file = file.replace(file_name, new_name)
    shutil.move(file, new_file)
