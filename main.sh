# python3 lenet31.py 1 2 2 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_2_2.log &
# python3 lenet31.py 1 3 2 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_3_2.log &
# python3 lenet31.py 1 4 2 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_4_2.log &
# python3 lenet31.py 1 5 2 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_5_2.log &
# python3 lenet31.py 1 1 3 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_1_3.log &
# python3 lenet31.py 1 2 4 0.01   > /opt/ml/disk/spnn/lenet31_alpha1_2_4.log &

# python lenet31.py 1 1 0.001 0.9  > /opt/ml/disk/spnn/lenet31_new/alpha1_1_0001_09 &
# python lenet31.py 1 1 0.001 0.8  > /opt/ml/disk/spnn/lenet31_new/alpha1_1_0001_08 &
# python lenet31.py 1 1 0.01  0.9  > /opt/ml/disk/spnn/lenet31_new/alpha1_1_001_09 &
# python lenet31.py 1 5 0.001 0.9  > /opt/ml/disk/spnn/lenet31_new/alpha1_5_0001_09 &
# python lenet31.py 0.01 5 0.001 0.9  > /opt/ml/disk/spnn/lenet31_new/alpha001_5_0001_09 &
# python lenet31.py 0.01 1 0.001 0.9  > /opt/ml/disk/spnn/lenet31_new/alpha001_1_0001_09 &

# python lenet5.py 1 1 0.001 0.9  > /opt/ml/disk/spnn/lenet5_new/alpha1_1_0001_09 &
# python lenet5.py 1 1 0.001 0.8  > /opt/ml/disk/spnn/lenet5_new/alpha1_1_0001_08 &
# python lenet5.py 1 1 0.01  0.9  > /opt/ml/disk/spnn/lenet5_new/alpha1_1_001_09 &
# python lenet5.py 1 5 0.001 0.9  > /opt/ml/disk/spnn/lenet5_new/alpha1_5_0001_09 &
# python lenet5.py 0.01 5 0.001 0.9  > /opt/ml/disk/spnn/lenet5_new/alpha001_5_0001_09 &
# python lenet5.py 0.01 1 0.001 0.9  > /opt/ml/disk/spnn/lenet5_new/alpha001_1_0001_09 &

# python3 lenet5.py 0.0001 5 2  > /opt/ml/disk/spnn/lenet5_5_2_alpha0001.log &
# python3 lenet5.py 0.001  5 2  > /opt/ml/disk/spnn/lenet5_5_2_alpha001.log &
# python3 lenet5.py 100    5 2  > /opt/ml/disk/spnn/lenet5_5_2_alpha100.log &
# python3 lenet5.py 5      5 2  > /opt/ml/disk/spnn/lenet5_5_5_alpha5.log &
# 
# python3 lenet5.py 0.0001 10 2  > /opt/ml/disk/spnn/lenet5_10_5_alpha0001.log &
# python3 lenet5.py 0.001  10 2  > /opt/ml/disk/spnn/lenet5_10_5_alpha001.log &
# python3 lenet5.py 5      10 2  > /opt/ml/disk/spnn/lenet5_10_5_alpha5.log &
# 
# python3 lenet5.py 0.0001 20 2  > /opt/ml/disk/spnn/lenet5_20_2_alpha0001.log &
# python3 lenet5.py 0.001  20 2  > /opt/ml/disk/spnn/lenet5_20_2_alpha001.log &
# python3 lenet5.py 5      20 2  > /opt/ml/disk/spnn/lenet5_20_2_alpha5.log &


# python3 pretrain.py 5 5 2    > /opt/ml/disk/spnn/vgg_pretrain.log &
# python vgg16.py 1 1 0.01 0.9 > /opt/ml/disk/spnn/vgg16_result/_1_1_1e-2_09.log &
# python vgg16.py 1 1 0.01 0.95 > /opt/ml/disk/spnn/vgg16_result/1_1_1e-2_095.log &
# python vgg16.py 1 1 0.001 0.9 > /opt/ml/disk/spnn/vgg16_result/1_1_1e-3_09.log &
# python vgg16.py 1 1 0.001 0.95 > /opt/ml/disk/spnn/vgg16_result/1_1_1e-3_095.log &

python3 pretrain_lenet31.py > /opt/ml/disk/spnn/log/pretrain_lenet31.log &
python3 pretrain_lenet5.py  > /opt/ml/disk/spnn/log/pretrain_lenet5.log &

# python lenet31_layerwise.py 1 1 0.0001 0.95 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-3_095 &
# python lenet31_layerwise.py 1 1 0.0001 0.90 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-3_090 &
# python lenet31_layerwise.py 1 1 0.001  0.95 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-2_095 &
# python lenet31_layerwise.py 1 1 0.001  0.90 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-2_090 &
# python lenet31_layerwise.py 1 1 0.0005  0.95 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-3_095 &
# python lenet31_layerwise.py 1 1 0.0005  0.90 > /opt/ml/disk/spnn/lenet31_layerwise/1_1_1e-3_090 &

chown -R 1041 *
chgrp -R 1041 *
#
