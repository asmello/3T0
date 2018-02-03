from keras.layers import Input, Dense, Conv2D, LeakyReLU, \
                         BatchNormalization, Flatten, add
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping

import numpy as np


class Estimator:

    def __init__(self, input_shape, output_dim, reg_const=1e-4, filepath=None):
        self.reg_const = reg_const
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.reg_const = reg_const
        self.model = load_model(filepath) if filepath else self._build_model()

    def save(self, filepath):
        self.model.save(filepath)

    def _conv_layer(self, x, filters, kernel_size):
        x = Conv2D(
              filters=filters,
              kernel_size=kernel_size,
              padding='same',
              use_bias=False,
              kernel_regularizer=l2(self.reg_const)
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def _res_layer(self, input_block, filters, kernel_size):
        x = self._conv_layer(input_block, filters, kernel_size)
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(self.reg_const)
        )(x)
        x = BatchNormalization()(x)
        x = add([input_block, x])
        x = LeakyReLU()(x)
        return x

    def _policy_head(self, x):
        x = Conv2D(
            filters=2,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(self.reg_const)
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            self.output_dim,
            use_bias=False,
            kernel_regularizer=l2(self.reg_const),
            activation='softmax',
            name='policy_head'
        )(x)
        return x

    def _value_head(self, x):
        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            kernel_regularizer=l2(self.reg_const)
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(
            20,
            use_bias=False,
            kernel_regularizer=l2(self.reg_const)
        )(x)
        x = LeakyReLU()(x)
        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=l2(self.reg_const),
            name='value_head'
        )(x)
        return x

    def _build_model(self):
        x_in = Input(shape=self.input_shape, name='input')
        x = self._conv_layer(x_in, 64, 3)
        x = self._res_layer(x, 64, 3)
        v_out = self._value_head(x)
        p_out = self._policy_head(x)
        model = Model(inputs=x_in, outputs=[p_out, v_out])
        model.compile(
            loss={
                'value_head': 'mean_squared_error',
                'policy_head': 'categorical_crossentropy'
            },
            optimizer='adam'
        )
        return model

    def compute(self, state, use_symmetry=True):
        '''Computes the priors and value of a random symmetry of the state.'''

        state = state.raw

        # get a random symmetry
        if use_symmetry:
            r, h, v = np.random.randint(0, 4), *np.random.randint(0, 2, 2)
            state = np.rot90(state, r)
            state = np.fliplr(state) if h else state
            state = np.flipud(state) if v else state

        # use the neural net to evaluate the state
        prior, value = self.model.predict(state[np.newaxis, ...])

        # inverse transformation
        if use_symmetry:
            prior = prior.reshape(self.input_shape[:2])
            prior = np.rot90(prior, -r)
            prior = np.fliplr(prior) if h else prior
            prior = np.flipud(prior) if v else prior
            prior = prior.flatten()

        return prior.squeeze(), value.squeeze()

    def update(self, games):
        '''Creates a new model by updating this one with playing history.'''

        # isolate inputs and outputs
        data = ((s, p, -s[0, 0, -1] * w) for h, w in games for s, p in h)
        x, p, v = zip(*data)

        # create a clone of this model
        new = Estimator(self.input_shape,
                        self.output_dim,
                        self.reg_const)
        new.model.set_weights(self.model.get_weights())

        # train the clone on these games
        new.model.fit(
            x=np.array(x),
            y=[np.array(p), np.array(v)],
            verbose=0,
            epochs=200,
            callbacks=[EarlyStopping('loss', 0.01, 10)]
        )

        return new
