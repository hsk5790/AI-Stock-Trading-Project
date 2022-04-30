import os
import threading
import numpy as np

<<<<<<< HEAD
<<<<<<< HEAD
# 얘는 main에 설정돼있다.
=======
>>>>>>> 1034ff0 (rltrader 코드 바로 사용할 수 있도록 수정했습니다.)
=======

>>>>>>> d1494d2 (아나콘다 프롬프트에서 실행했을 때 바로 작동할 수 있도록 코드 수정해 놨습니다. conda activate bkst 해서 bkst 가상환경에서 실행해주세요.)
if os.environ.get('KERAS_BACKEND', 'tensorflow') == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, \
        BatchNormalization, Dropout, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras import backend
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv1D, \
        BatchNormalization, Dropout, MaxPooling1D, Flatten
    from keras.optimizers import SGD


# Network class : 밑에 있는 DNN, LSTMNetwork, CNN class의 상위 클래스로 사용된다.(이 class 자체로 사용하지는 않는다.)
# 신경망이 공통으로 가질 attribute와 function()을 정의해 놓은 class다.
class Network:
    # A3C에서는 thread를 이용해 병렬로 신경망을 사용하기 때문에 thread간의 충돌 방지를 위해 Lock 클래스 object를 가지고 있다.
    # A3C는 강화학습 알고리즘이다.
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr                          # 신경망의 학습 속도
        self.shared_network = shared_network  # 신경망의 상단부로 여러 신경망이 공유할 수 있다.
        self.activation = activation          # 신경망의 출력층 activation function. ex) 'sigmoid', 'linear'
        self.loss = loss                      # 신경망의 손실 함수
        self.model = None                     # keras 라이브러리로 구성한 최종 신경망 모델

    # 신경망을 통해 투자 행동별 가치나 확률 계산
    def predict(self, sample):
        # with는 에러가 나든 안 나든 마지막에 무조건 close() 해주는 녀석
        with self.lock:
            pred = self.model.predict_on_batch(sample).flatten()
            return pred

    # 배치 학습을 위한 데이터 생성
    # 학습 데이터와 label x,y를 입력으로 받아서 모델을 학습시킨다.
    def train_on_batch(self, x, y):
        loss = 0.
        # with는 에러가 나든 안 나든 마지막에 무조건 close() 해주는 녀석
        with self.lock:
            history = self.model.fit(x, y, epochs=10, verbose=False)  # verbose: 상세하게 log를 출력할지 안 할지.
            loss += np.sum(history.history['loss'])
        return loss

    # 학습한 신경망을 파일로 저장
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    # 파일로 저장한 신경망을 로드
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    # 신경망의 상단부(DNN, LSTM, CNN 신경망의 공유 신경망)를 생성하는 클래스 함수
    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        # output_dim은 pytorch에서 필요
        if net == 'dnn':
            return DNN.get_network_head(Input((input_dim,)))
        elif net == 'lstm':
            return LSTMNetwork.get_network_head(Input((num_steps, input_dim)))
        elif net == 'cnn':
            return CNN.get_network_head(Input((num_steps, input_dim)))
    

class DNN(Network):
    # DNN 클래스의 생성자에서 공유 신경망이 지정돼 있지 않으면 스스로 생성한다.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.input_dim,))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation, 
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    # DNN 클래스가 어떠한 구조로 신경망을 구축하는지 알 수 있다.
    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', 
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)               # 배치 정규화 => 학습 안정화
        output = Dropout(0.1)(output)                       # Dropout => overfitting을 부분적으로 피한다.
        output = Dense(128, activation='sigmoid',           # 은닉층 af : sigmoid
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid', 
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    # 학습 데이터나 샘플의 shape을 변경하고 상위 클래스의 함수를 그대로 호출한다.
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    # 학습 데이터나 샘플의 shape을 변경하고 상위 클래스의 함수를 그대로 호출한다.
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)
    

# DNN과 전체적으로 비슷하지만 attribute의 차이로 num_steps 변수를 가지고 있다.
# train_on_batch() 함수와 predict() 함수에서 학습 데이터와 샘플의 shape을 변경할 때 num_steps 변수를 사용한다.
class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation, 
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    # 이 함수를 수정해 LSTM 신경망의 구조를 조정할 수 있다.
    # LSTM layer를 여러 겹 쌓을 경우 마지막 LSTM layer를 제외하고 return_sequences=True 인자를 줘야한다.
    # return_sequences=True를 주면 해당 layer의 출력 개수를 num_steps 만큼 유지한다.
    @staticmethod
    def get_network_head(inp):
        # cuDNN 사용을 위한 조건
        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
        output = LSTM(256, dropout=0.1, return_sequences=True,
                    kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1, return_sequences=True,
                    kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1, return_sequences=True,
                    kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1, kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    # train_on_batch() 함수와 predict() 함수에서 학습 데이터와 샘플의 shape을 변경할 때 num_steps 변수를 사용하고
    # 상위 클래스인 Network 클래스의 함수를 호출한다.
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)


# CNN 신경망은 LSTM 신경망과 마찬가지로 다차원 데이터를 다룰 수 있다.
# convolution layer는 1D, 2D, 3D 등으로 다양한 차원을 다룰 수 있다.
# 책에서는 2D를 사용했는데 현재 코드는 1D를 사용하고 있다. 이유는?
# num_steps로 차원 크기를 조정한다.
class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_steps = num_steps
        inp = None
        output = None
        if self.shared_network is None:
            inp = Input((self.num_steps, self.input_dim, 1))
            output = self.get_network_head(inp).output
        else:
            inp = self.shared_network.input
            output = self.shared_network.output
        output = Dense(
            self.output_dim, activation=self.activation,
            kernel_initializer='random_normal')(output)
        self.model = Model(inp, output)
        self.model.compile(
            optimizer=SGD(learning_rate=self.lr), loss=self.loss)

    # padding='same'을 줘서 입력과 출력의 크기가 같게 설정
    # 합성곱 window 크기로 사용되는 kernel_size=5로 설정
    # CNN의 경우 가변점이 많기 때문에 보다 더 다양한 파라미터로 실험을 해야한다.
    @staticmethod
    def get_network_head(inp):
        output = Conv1D(256, kernel_size=5,
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling1D(pool_size=2, padding='same')(output)
        output = Dropout(0.1)(output)
        output = Conv1D(64, kernel_size=5,
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling1D(pool_size=2, padding='same')(output)
        output = Dropout(0.1)(output)
        output = Conv1D(32, kernel_size=5,
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling1D(pool_size=2, padding='same')(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    # CNN의 경우 일반적으로 이미지 데이터를 취급하기 때문에 마지막 차원의 경우 RGB와 같은 이미지 channel이 들어간다.
    # 그러나 주식데이터는 channel이라 할게 없기 때문에 1로 고정한다.
    # 위의 다른 신경망과 마찬가지로 학습 데이터와 샘플의 모양을 바꾸고 Network 클래스의 함수를 그대로 호출한다.
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)
