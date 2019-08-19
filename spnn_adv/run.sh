python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch1 -e 1e-3 --epoch 1
python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch2 -e 1e-3 --epoch 2
python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch4 -e 1e-3 --epoch 4
python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch8 -e 1e-3 --epoch 8
python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch16 -e 1e-3 --epoch 16
python3 main_newshort.py vgg16 CIFAR10 ~/data/ pretrain/vgg16_10_cifar10.pk 0.911 -p checkpoint/adv_vgg16_epoch32 -e 1e-3 --epoch 32

