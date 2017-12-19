import tensorflow.contrib.keras as K
import numpy as np


def gen(batch_size=25):
    while True:
        X = (np.random.random([batch_size,224,224,3])-0.5)*127
        Y = np.random.randint(0, 1, [batch_size,1000])
        yield X,Y

gen_tr = gen()
next(gen_tr)

base_net = K.applications.resnet50.ResNet50
preprocess_input = K.applications.resnet50.preprocess_input


base_model = base_net(weights='imagenet')
base_model.summary()

base_model.compile(
        optimizer = K.optimizers.Adadelta(lr=0.01, decay=5e-5),
        loss = 'binary_crossentropy',
        metrics=['binary_crossentropy']
        )

base_model.fit_generator(gen_tr, 400, 100)
