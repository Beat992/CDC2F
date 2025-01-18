import os

path = r''
phase = ['train', 'val', 'test']
for ph in phase:
    file_path = os.path.join(path, '{}/A'.format(ph))
    path_list = os.listdir(file_path)
    save_dir = os.path.join(path, 'txt')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_name = []

    for name in path_list:
        path_name.append(name)

    for file_name in path_name:
        # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有"save.txt"会自动创建
        with open(save_dir + '/{}.txt'.format(ph), 'a') as file:
            file.write('A/' + file_name + ' ' +
                       'B/' + file_name + ' ' +
                       'mask/' + file_name[:-4] + '.png' +
                       '\n')
        file.close()
