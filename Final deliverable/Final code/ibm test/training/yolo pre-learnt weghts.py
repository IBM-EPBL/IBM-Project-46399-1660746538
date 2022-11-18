from utils import WeightReader, decode_netout, draw_boxes
wt_path = "yolov3.weights"
weight_reader = WeightReader(wt_path)
weight_reader
Out[10]:
In [11]:
#This is doing the weight initialisation from random to the weights that have already being learnt by the YOLO model we are implementing

weight_reader.reset()

#we need to adjust the nb_conv because we removed the other layer they had and added our own, and because it's our own
#it won't have learnt weight

nb_conv = 22

for i in range(1, nb_conv+1):
    conv_layer = model.get_layer('conv_' + str(i))
    
    if i < nb_conv:
        norm_layer = model.get_layer('norm_' + str(i))
        
        size = np.prod(norm_layer.get_weights()[0].shape)

        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)

        weights = norm_layer.set_weights([gamma, beta, mean, var])       
        
    if len(conv_layer.get_weights()) > 1:
        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])
    else:
        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])

#model.fit(X, y, batch_size=32, epochs=50, validation_data=(X_test,y_test),  callbacks = [early_stop,tensorboard])

model.fit(X, y, batch_size=32, epochs=150, validation_data=(X_test,y_test))
Train on 1148 samples, validate on 42 samples
Epoch 1/150
1148/1148 [==============================] - 22s 19ms/step - loss: 3.7357 - mean_absolute_error: 0.9446 - val_loss: 0.1210 - val_mean_absolute_error: 0.2888
Epoch 2/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.1705 - mean_absolute_error: 0.3298 - val_loss: 0.1227 - val_mean_absolute_error: 0.3113
Epoch 3/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0956 - mean_absolute_error: 0.2457 - val_loss: 0.1558 - val_mean_absolute_error: 0.3457
Epoch 4/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0568 - mean_absolute_error: 0.1875 - val_loss: 0.0711 - val_mean_absolute_error: 0.2322
Epoch 5/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0488 - mean_absolute_error: 0.1741 - val_loss: 0.0767 - val_mean_absolute_error: 0.2327
Epoch 6/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0325 - mean_absolute_error: 0.1399 - val_loss: 0.0329 - val_mean_absolute_error: 0.1422
Epoch 7/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0270 - mean_absolute_error: 0.1269 - val_loss: 0.0249 - val_mean_absolute_error: 0.1341
Epoch 8/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0265 - mean_absolute_error: 0.1263 - val_loss: 0.0501 - val_mean_absolute_error: 0.2044
Epoch 9/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0206 - mean_absolute_error: 0.1090 - val_loss: 0.0332 - val_mean_absolute_error: 0.1440
Epoch 10/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0191 - mean_absolute_error: 0.1048 - val_loss: 0.0597 - val_mean_absolute_error: 0.2215
Epoch 11/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0185 - mean_absolute_error: 0.1037 - val_loss: 0.0508 - val_mean_absolute_error: 0.1960
Epoch 12/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0167 - mean_absolute_error: 0.0977 - val_loss: 0.0366 - val_mean_absolute_error: 0.1563
Epoch 13/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0158 - mean_absolute_error: 0.0959 - val_loss: 0.0987 - val_mean_absolute_error: 0.2384
Epoch 14/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0147 - mean_absolute_error: 0.0918 - val_loss: 0.0177 - val_mean_absolute_error: 0.1052
Epoch 15/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0138 - mean_absolute_error: 0.0894 - val_loss: 0.0111 - val_mean_absolute_error: 0.0867
Epoch 16/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0121 - mean_absolute_error: 0.0829 - val_loss: 0.0285 - val_mean_absolute_error: 0.1306
Epoch 17/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0117 - mean_absolute_error: 0.0828 - val_loss: 0.0529 - val_mean_absolute_error: 0.1700
Epoch 18/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0097 - mean_absolute_error: 0.0735 - val_loss: 0.0082 - val_mean_absolute_error: 0.0775
Epoch 19/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0097 - mean_absolute_error: 0.0740 - val_loss: 0.0114 - val_mean_absolute_error: 0.0885
Epoch 20/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0088 - mean_absolute_error: 0.0697 - val_loss: 0.0044 - val_mean_absolute_error: 0.0537
Epoch 21/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0088 - mean_absolute_error: 0.0709 - val_loss: 0.0078 - val_mean_absolute_error: 0.0665
Epoch 22/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0077 - mean_absolute_error: 0.0657 - val_loss: 0.0121 - val_mean_absolute_error: 0.0970
Epoch 23/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0078 - mean_absolute_error: 0.0665 - val_loss: 0.0080 - val_mean_absolute_error: 0.0703
Epoch 24/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0078 - mean_absolute_error: 0.0671 - val_loss: 0.0060 - val_mean_absolute_error: 0.0563
Epoch 25/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0070 - mean_absolute_error: 0.0620 - val_loss: 0.0043 - val_mean_absolute_error: 0.0510
Epoch 26/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0066 - mean_absolute_error: 0.0613 - val_loss: 0.0042 - val_mean_absolute_error: 0.0479
Epoch 27/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0057 - mean_absolute_error: 0.0562 - val_loss: 0.0054 - val_mean_absolute_error: 0.0546
Epoch 28/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0058 - mean_absolute_error: 0.0576 - val_loss: 0.0079 - val_mean_absolute_error: 0.0748
Epoch 29/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0055 - mean_absolute_error: 0.0556 - val_loss: 0.0053 - val_mean_absolute_error: 0.0555
Epoch 30/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0048 - mean_absolute_error: 0.0525 - val_loss: 0.0046 - val_mean_absolute_error: 0.0494
Epoch 31/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0057 - mean_absolute_error: 0.0579 - val_loss: 0.0041 - val_mean_absolute_error: 0.0506
Epoch 32/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0050 - mean_absolute_error: 0.0534 - val_loss: 0.0101 - val_mean_absolute_error: 0.0840
Epoch 33/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0047 - mean_absolute_error: 0.0521 - val_loss: 0.0053 - val_mean_absolute_error: 0.0532
Epoch 34/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0044 - mean_absolute_error: 0.0495 - val_loss: 0.0037 - val_mean_absolute_error: 0.0462
Epoch 35/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0040 - mean_absolute_error: 0.0484 - val_loss: 0.0074 - val_mean_absolute_error: 0.0650
Epoch 36/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0040 - mean_absolute_error: 0.0484 - val_loss: 0.0087 - val_mean_absolute_error: 0.0727
Epoch 37/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0038 - mean_absolute_error: 0.0467 - val_loss: 0.0129 - val_mean_absolute_error: 0.0981
Epoch 38/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0044 - mean_absolute_error: 0.0503 - val_loss: 0.0048 - val_mean_absolute_error: 0.0517
Epoch 39/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0035 - mean_absolute_error: 0.0454 - val_loss: 0.0067 - val_mean_absolute_error: 0.0580
Epoch 40/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0031 - mean_absolute_error: 0.0421 - val_loss: 0.0073 - val_mean_absolute_error: 0.0648
Epoch 41/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0029 - mean_absolute_error: 0.0412 - val_loss: 0.0046 - val_mean_absolute_error: 0.0514
Epoch 42/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0028 - mean_absolute_error: 0.0405 - val_loss: 0.0037 - val_mean_absolute_error: 0.0425
Epoch 43/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0031 - mean_absolute_error: 0.0426 - val_loss: 0.0050 - val_mean_absolute_error: 0.0563
Epoch 44/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0029 - mean_absolute_error: 0.0415 - val_loss: 0.0056 - val_mean_absolute_error: 0.0590
Epoch 45/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0026 - mean_absolute_error: 0.0392 - val_loss: 0.0063 - val_mean_absolute_error: 0.0628
Epoch 46/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0025 - mean_absolute_error: 0.0383 - val_loss: 0.0045 - val_mean_absolute_error: 0.0500
Epoch 47/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0031 - mean_absolute_error: 0.0421 - val_loss: 0.0071 - val_mean_absolute_error: 0.0620
Epoch 48/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0028 - mean_absolute_error: 0.0409 - val_loss: 0.0054 - val_mean_absolute_error: 0.0571
Epoch 49/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0026 - mean_absolute_error: 0.0391 - val_loss: 0.0056 - val_mean_absolute_error: 0.0549
Epoch 50/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0022 - mean_absolute_error: 0.0355 - val_loss: 0.0038 - val_mean_absolute_error: 0.0466
Epoch 51/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0352 - val_loss: 0.0070 - val_mean_absolute_error: 0.0664
Epoch 52/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0019 - mean_absolute_error: 0.0340 - val_loss: 0.0034 - val_mean_absolute_error: 0.0419
Epoch 53/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0348 - val_loss: 0.0074 - val_mean_absolute_error: 0.0663
Epoch 54/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0022 - mean_absolute_error: 0.0362 - val_loss: 0.0032 - val_mean_absolute_error: 0.0397
Epoch 55/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0356 - val_loss: 0.0058 - val_mean_absolute_error: 0.0588
Epoch 56/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0360 - val_loss: 0.0040 - val_mean_absolute_error: 0.0494
Epoch 57/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0351 - val_loss: 0.0079 - val_mean_absolute_error: 0.0670
Epoch 58/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0019 - mean_absolute_error: 0.0333 - val_loss: 0.0035 - val_mean_absolute_error: 0.0451
Epoch 59/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0017 - mean_absolute_error: 0.0321 - val_loss: 0.0074 - val_mean_absolute_error: 0.0636
Epoch 60/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0016 - mean_absolute_error: 0.0314 - val_loss: 0.0079 - val_mean_absolute_error: 0.0718
Epoch 61/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0021 - mean_absolute_error: 0.0343 - val_loss: 0.0061 - val_mean_absolute_error: 0.0576
Epoch 62/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0019 - mean_absolute_error: 0.0340 - val_loss: 0.0040 - val_mean_absolute_error: 0.0489
Epoch 63/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0015 - mean_absolute_error: 0.0297 - val_loss: 0.0075 - val_mean_absolute_error: 0.0688
Epoch 64/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0016 - mean_absolute_error: 0.0303 - val_loss: 0.0053 - val_mean_absolute_error: 0.0559
Epoch 65/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0018 - mean_absolute_error: 0.0325 - val_loss: 0.0065 - val_mean_absolute_error: 0.0661
Epoch 66/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0018 - mean_absolute_error: 0.0329 - val_loss: 0.0047 - val_mean_absolute_error: 0.0523
Epoch 67/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0296 - val_loss: 0.0075 - val_mean_absolute_error: 0.0625
Epoch 68/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0283 - val_loss: 0.0052 - val_mean_absolute_error: 0.0581
Epoch 69/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0016 - mean_absolute_error: 0.0308 - val_loss: 0.0058 - val_mean_absolute_error: 0.0579
Epoch 70/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0298 - val_loss: 0.0043 - val_mean_absolute_error: 0.0488
Epoch 71/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0288 - val_loss: 0.0041 - val_mean_absolute_error: 0.0470
Epoch 72/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0013 - mean_absolute_error: 0.0283 - val_loss: 0.0059 - val_mean_absolute_error: 0.0566
Epoch 73/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0011 - mean_absolute_error: 0.0262 - val_loss: 0.0042 - val_mean_absolute_error: 0.0481
Epoch 74/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0012 - mean_absolute_error: 0.0271 - val_loss: 0.0054 - val_mean_absolute_error: 0.0569
Epoch 75/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0012 - mean_absolute_error: 0.0269 - val_loss: 0.0068 - val_mean_absolute_error: 0.0634
Epoch 76/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0012 - mean_absolute_error: 0.0268 - val_loss: 0.0047 - val_mean_absolute_error: 0.0506
Epoch 77/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0287 - val_loss: 0.0052 - val_mean_absolute_error: 0.0499
Epoch 78/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0014 - mean_absolute_error: 0.0288 - val_loss: 0.0033 - val_mean_absolute_error: 0.0423
Epoch 79/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0013 - mean_absolute_error: 0.0274 - val_loss: 0.0044 - val_mean_absolute_error: 0.0498
Epoch 80/150
1148/1148 [==============================] - 16s 14ms/step - loss: 0.0011 - mean_absolute_error: 0.0259 - val_loss: 0.0046 - val_mean_absolute_error: 0.0519
    return a[2] * a[3] + b[2] * b[3] - intersection(a, b)

def iou(y_true, y_pred):
    return intersection(y_true, y_pred) / union(y_true, y_pred)
In [23]:
all_iou = np.zeros((np.shape(y)[0],))
for i in range(40): # np.shape(X)[0]): 
    ye = model.predict(X[i:i+1, :, :, :])
    plot_example(X[i,:,:,:], ye[0, :]*192)
    all_iou[i] = iou(y[i, :], ye[0, :])
