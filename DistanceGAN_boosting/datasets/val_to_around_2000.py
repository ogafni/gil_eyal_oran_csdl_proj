import os
import random
import shutil

relative_path = '.' # Change if not in dataset folder
# task = 'edges2shoes' # edges2shoes or edges2handbags
task = 'edges2handbags'



if task == 'edges2handbags':
    task_path = os.path.join(relative_path,'edges2handbags')
    original_train_num = 138567 #handbags = 138567 ; shoes = 49825
elif task == 'edges2shoes':
    task_path = os.path.join(relative_path,'edges2shoes')
    original_train_num = 49825 #handbags = 138567 ; shoes = 49825
else:    
    print('Wrong task name')

n_new_val = 1800

train_path =os.path.join(task_path,'train')
val_path =os.path.join(task_path,'val')
file_names = os.listdir(train_path)
n_train = len(file_names)

if n_train==original_train_num:
    random.seed(1234)
    rand_list = random.sample(file_names,n_new_val)
    for file_name in rand_list:
        shutil.move(os.path.join(train_path,file_name),os.path.join(val_path,file_name))
    print('Done moving files in ', task)
else:
    print('Wrong number of files, did you already move them?')

print(str(len(os.listdir(val_path)))+' files in val folder')
