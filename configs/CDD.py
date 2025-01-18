import os

# <---------usually modify----------->
resume = False
num_epochs = 200
init_learning_rate = 2e-3
weight_decay = 5e-2
train_batch_size = 16  # resnet18 32, 50: 8

model_name = 'our'
dataset_name = 'CDD'
note = ''
threshold = 0.5

# <--------dont modify-------- >
save_name = f'{dataset_name}_{model_name}.pth'
channel_resnet18 = 1024       # 512 + 256 + 128 + 64 + 64
channel_resnet18_s4 = 512
channel_resnet50 = 3904
channel_resnet50_s4 = 3904

momentum = 0.90
val_batch_size = 1

# t0_mean, t0_std = [0.46509804, 0.46042014, 0.39211731], [0.1854678, 0.17633663, 0.1648103]
# t1_mean, t1_std = [0.36640143, 0.35843183, 0.31020384], [0.15798439, 0.15536726, 0.14649872]

# <--------path config--------->
base_path = ''
data_path = ''

train_data_path = os.path.join(data_path, 'train')
train_label_path = os.path.join(data_path, 'train')
train_txt_path = os.path.join(data_path, 'txt/train_10000.txt')  # 修改

val_data_path = os.path.join(data_path, 'val')
val_label_path = os.path.join(data_path, 'val')
val_txt_path = os.path.join(data_path, 'txt/val_3000.txt')

test_data_path = os.path.join(data_path, 'test')
test_label_path = os.path.join(data_path, 'test')
test_txt_path = os.path.join(data_path, 'txt/test_3000.txt')

training_best_ckpt = os.path.join(base_path, 'checkpoints')
if not os.path.exists(training_best_ckpt):
    os.mkdir(training_best_ckpt)

save_path = os.path.join(base_path, 'result')
if not os.path.exists(save_path):
    os.mkdir(save_path)

pretrain_model_path_ResNet18 = os.path.join(base_path, 'pretrained_weight/resnet18-5c106cde.pth')
pretrain_model_path_ResNet34 = os.path.join(base_path, 'pretrained_weight/resnet34-333f7ec4.pth')
pretrain_model_path_ResNet50 = os.path.join(base_path, 'pretrained_weight/resnet50-19c8e357.pth')
