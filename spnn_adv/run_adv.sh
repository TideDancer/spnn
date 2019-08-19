# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch1 FGSM > log/adv_vgg16_epoch1_FGSM.log
# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch1 PGD  > log/adv_vgg16_epoch1_PGD.log
# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch2 FGSM > log/adv_vgg16_epoch2_FGSM.log
# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch2 PGD  > log/adv_vgg16_epoch2_PGD.log
# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch4 FGSM > log/adv_vgg16_epoch4_FGSM.log
# python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch4 PGD  > log/adv_vgg16_epoch4_PGD.log
python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch8 FGSM  > log/adv_vgg16_epoch8_FGSM.log
python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch8 PGD   > log/adv_vgg16_epoch8_PGD.log
python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch16 FGSM > log/adv_vgg16_epoch16_FGSM.log
python3 attack_verify.py CIFAR10 checkpoint/adv_vgg16_epoch16 PGD  > log/adv_vgg16_epoch16_PGD.log

