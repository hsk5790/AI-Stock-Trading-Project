import os                   # 폴더 생성, 파일 경로 준비
import logging              # 학습 과정 중에 정보를 기록하기 위해 사용
import abc                  # 추상 클래스를 정의하기 위해 사용
import collections
import threading
import time                 # 학습 시간 측정
import json
import numpy as np
from tqdm import tqdm
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent
from quantylab.rltrader.networks import Network, DNN, LSTMNetwork, CNN
from quantylab.rltrader.visualizer import Visualizer
from quantylab.rltrader import utils
from quantylab.rltrader import settings


logger = logging.getLogger(settings.LOGGER_NAME)


# DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner 클래스가 상속하는 상위 클래스다.
class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self
                 , rl_method='rl'               # 강화학습 기법 => 하위 클래스에 따라 달라진다.
                 , stock_code=None              # 강화학습을 진행하는 주식 종목 코드
                 , chart_data=None              # 주식 종목의 일봉 차트 데이터
                 , training_data=None           # 학습을 위한 전처리된 학습 데이터
                 , min_trading_price=100000     # 투자 최소 단위
                 , max_trading_price=10000000   # 투자 최대 단위
                 , net='dnn'                    # 신경망 종류
                 , num_steps=1                  # LSTM, CNN 신경망에서 사용하는 샘플 묶음 크기
                 , lr=0.0005                    # 학습 속도 => 이 값이 너무 크면 학습이 제대로 진행되지 않는다.
                 , discount_factor=0.9
                 , num_epoches=1000
                 , balance=100000000
                 , start_epsilon=1
                 , value_network=None           # 가치 신경망
                 , policy_network=None          # 정책 신경망
                 , output_path=''               # 학습 과정에서 발생하는 로그, 가시화 결과 및 학습 종료 후 저장되는 신경망 모델 파일을 지정된 경로에 저장
                 , reuse_models=True
                 ):
        # 인자 확인
        # assert는 뒤의 조건이 True가 아니면 AssertError를 발생시킨다.
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)      # chart_data를 인자로 하여 Environment 클래스의 instance를 생성한다.
                                                        # environment는 차트 데이터를 순서대로 읽으면서 주가, 거래량 등의 데이터를 제공한다.
        # 에이전트 설정
        # 이 강화학습 environment를 인자로 Agent 클래스의 instance를 생성한다.
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        # 학습 데이터
        # 학습에 사용할 feature들 포함
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models        # 모델이 이미 존재하는 경우 재활용한다.
        # 가시화 모듈
        self.visualizer = Visualizer()  # 학습 과정을 가시화하기 위해 Visualizer 클래스 객체 생성
        # 메모리
        # 강화학습 과정에서 발생하는 각종 데이터를 쌓아두기 위한 변수들 생성
        self.memory_sample = []     # 학습 데이터 샘플
        self.memory_action = []     # 수행한 행동
        self.memory_reward = []     # 획득한 보상
        self.memory_value = []      # 행동의 예측 가치
        self.memory_policy = []     # 행동의 예측 확률
        self.memory_pv = []         # 포트폴리오 가치
        self.memory_num_stocks = [] # 보유 주식 수
        self.memory_exp_idx = []    # 탐험 위치
        # 에포크 관련 정보
        self.loss = 0.              # 발생한 손실
        self.itr_cnt = 0            # 수익 발생 횟수
        self.exploration_cnt = 0    # 탐험 횟수
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path  # 로그, 가시화, 학습 모델 등은 output_path로 지정된 경로 하위에 저장된다.

    # init_value_network() : net에 지정된 신경망 종류에 맞게 가치 신경망을 생성한다.
    # 가치 신경망은 손익률을 회귀분석하는 모델. 그래서 af은 linear로, 손실 함수는 MSE로 설정
    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        # reuse_models가 True이고 value_network_path에 지정된 경로에 신경망 모델 파일이 존재하면 신경망 모델 파일을 불러온다.
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    # init_policy_network() : 정책 신경망 생성 함수
    # 가치 신경망과 유사하지만 af를 sigmoid, 손실 함수를 binary_crossentropy를 사용했다.
    # 정책 신경망은 샘플에 대해서 PV를 높이는 방향으로 행동하는 분류 모델이다.
    # af를 sigmoid를 써서 결과값이 0~1 사이의 확률로 사용한다.
    def init_policy_network(self, shared_network=None, activation='sigmoid', 
                            loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        # reuse_models가 True이고 value_network_path에 지정된 경로에 신경망 모델 파일이 존재하면 신경망 모델 파일을 불러온다.
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

    # reset() : 에포크 초기화 함수
    # 학습 데이터를 처음부터 읽기 위해 training_data_idx = -1로 재설정합니다.
    def reset(self):
        self.sample = None          # 읽어온 데이터는 sample에 저장되는데, 초기화 단계에는 읽어온 학습 데이터가 없기 때문에 None으로 할당한다
        self.training_data_idx = -1 # 이 값은 학습 데이터를 읽어가면서 1씩 증가한다.
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.              # 신경망의 결과가 학습 데이터와 얼마나 차이가 있는지를 저장하는 변수다.(줄어들면 좋다)
        self.itr_cnt = 0            # 수행한 epoch 수 저장
        self.exploration_cnt = 0    # 무작위 투자를 수행한 횟수 저장(epsilon이 0.1이고 100번의 투자 결정이 있아면 약 10번의 무작위 투자를 한 것이다.)
        self.batch_size = 0

    # build_sample() : environment object에서 샘플을 획득하는 함수
    def build_sample(self):
        self.environment.observe()  # 환경 객체의 ovserve() 함수를 호출해 차트 데이터의 현재 index에서 다음 index를 읽게 한다.
        if len(self.training_data) > self.training_data_idx + 1:    # 다음 index가 존재하는지 확인
            self.training_data_idx += 1                             # 다음 index가 존재하면 training_data_idx 변수를 1 증가시킨다.
            # training_data에서 train_data_idx의 데이터를 받아와서 sample로 저장한다.
            self.sample = self.training_data.iloc[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None

    # get_batch() : 신경망을 학습히가 위해 배치 학습 데이터를 생성하는 함수
    # get_batch() 함수는 추상 메소드로서 ReinforcementLearner 클래스의 하위 클래스들은 반드시 이 함수를 구현해야 한다.
    # ReinforcementLearner 클래스를 상속하고도 이 추상 메소드를 구현하지 implement하지 않으면 NotImplemented 예외가 발생한다.
    @abc.abstractmethod
    def get_batch(self):
        pass

    # fit() : 가치 신경망 및 정책 신경망 학습 요청 함수
    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            self.loss = loss

    # visualize() : 에포크 정보 가시화 함수
    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action    # agent의 행동
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks            # 보유 주식 수 (tip: 리스트에 곱하기를 하면 뒤에 똑같은 리스트를 붙여준다.)
        if self.value_network is not None:                                                      # 가치 신경망 출력
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_value                      # 포트폴리오 가치는 환경의 일봉 수보다 (num_steps -1) 만큼 부족해서 쓴 코드
        if self.policy_network is not None:                                                     # 정책 신경망 출력
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv   # 포트폴리오 가치
        # 객체 visualizer의 plot() 함수 호출 => 생성된 epoch 결과 그림을 PNG 그림 파일로 저장한다.
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS, 
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))

    # run() : 강화학습 수행 함수
    # ReinforcementLearner 클래스의 핵심 함수다.
    def run(self, learning=True):
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} '
            f'LR:{self.lr} DF:{self.discount_factor} ' # discount_factor는 먼 과거의 행동일수록 현재의 보상을 약하게 적용한다.
        )
        # with는 에러가 나든 안 나든 마지막에 무조건 close() 해주는 녀석
        with self.lock:
            logger.debug(info)

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        # visualizer의 prepare() 함수를 호출해 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f)) # 이미 폴더에 저장된 파일이 있을 경우 모두 삭제

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0 # 수행한 epoch 중에서 가장 높은 포트폴리오 가치(PV)가 저장된다.
        epoch_win_cnt = 0       # 수행한 epoch 중에서 수익이 발생한 epoch 수를 저장한다.

        # 지정된 에포크 수 만큼 주식투자 시뮬레이션을 반복하며 학습하는 반복문
        for epoch in tqdm(range(self.num_epoches)):
            # epoch의 시작 시간 기록
            time_start_epoch = time.time()

            # num_step 만큼 샘플을 만들기 위한 Queue를 초기화
            # Queue는 선입선출(First In First Out) 자료구조이다.
            # Deque는 양방향 Queue 자료구조다.
            # maxlen parameter를 주면 이 양방향 Queue의 크기를 제한할 수 있다.
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 epsilon은 star_epsilon에서 점차 줄어든다.
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon

            # tqdm의 leave parameter는 상태만 표시
            for i in tqdm(range(len(self.training_data)), leave=False):
                # build_sample() 함수를 호출해 환경 객체로부터 하나의 샘플을 읽어온다.
                next_sample = self.build_sample()
                if next_sample is None: # next_sample이 None이라면 마지막까지 데이터를 다 읽은 것이므로 반복문을 종료한다.
                    break # 반복문 종료

                # num_steps의 개수만큼 샘플이 준비돼야 행동을 결정할 수 있기 때문에 샘플 Queue에 샘플이 모두 찰 때까지
                # continue를 통해 이후 logic을 건너뛴다.
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                # 각 신경망 객체의 predict() 함수를 호출해 예측 행동 가치와 예측 행동 확률을 구한다.
                # 이렇게 구한 가치와 확률로 투자 행동을 결정한다.
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 신경망 또는 탐험에 의한 행동 결정
                # 여기서는 매도와 매수 중 하나를 결정한다.
                # 이 행동 결정은 무작위 투자 비율인 epsilon 값의 확률로 하거나, 그렇지 않은 경우 신경망의 출력을 통해 결정된다.
                # 정책 신경망의 출력은 매수를 했을 때와 매도를 했을 때의 포트폴리오 가치를 높일 확률을 의미한다.
                # 가치 신경망의 출력은 행동에 대한 예측 가치(손익률)을 의미한다.
                # decide_action() 함수가 반환하는 값은 1. 결정한 행동인 action, 2. 결정에 대한 확신도인 confidence, 3. 무작위 투자 유무인 exploration
                action, confidence, exploration = \
                    self.agent.decide_action(pred_value, pred_policy, epsilon)

                # agent의 act()함수는 결정한 행동을 수행하고 보상을 획득한다.
                reward = self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 memory로 시작하는 변수들에 저장한다.
                # memory 변수들은 두 가지 목적으로 사용한다.
                # 1. 학습에서 배치 학습 데이터로 사용, 2. visualizer에서 차트를 그릴 때 사용한다.
                self.memory_sample.append(list(q_sample))               # 학습 데이터의 샘플
                self.memory_action.append(action)                       # agent의 행동
                self.memory_reward.append(reward)                       # 보상
                if self.value_network is not None:
                    self.memory_value.append(pred_value)                # 가치 신경망 출력
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)              # 정책 신경망 출력
                self.memory_pv.append(self.agent.portfolio_value)       # 포트폴리오 가치
                self.memory_num_stocks.append(self.agent.num_stocks)    # 보유 주식 수
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)  # 탐험 위치 저장하는 배열


                # 반복에 대한 정보 갱신
                self.batch_size += 1        # 배치 크기
                self.itr_cnt += 1           # 반복 카운팅 횟수
                self.exploration_cnt += 1 if exploration else 0     # 무작위 투자 횟수 exploration_cnt를 증가시킨다
                                                                    # exploration_cnt의 경우 탐험한 경우에만 1을 증가, 아니면 0을 더해서 변화가 없게 한다.

            # 에포크 종료 후 학습
            if learning:
                self.fit()

            # 하나의 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(self.num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0')    # str().rjust() : 원하는 문자를 앞에 채워준다. 180pg 참고
            time_end_epoch = time.time()                                            # 현재 시간
            elapsed_time_epoch = time_end_epoch - time_start_epoch                  # epoch 수행 소요 시간
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '  
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')    # loss 변수는 epoch동안 수행한 미니 배치들의 학습 손실을 모두 더해놓은 상태다.
                                                                        # loss를 학습 횟수만큼 나눠서 미니 배치의 평균 학습 손실로 갱신한다.

            # visualize() 함수를 이용해 하나의 에포크 관련 정보 가시화
            if self.num_epoches == 1 or (epoch + 1) % int(self.num_epoches / 10) == 0:
                self.visualize(epoch_str, self.num_epoches, epsilon)

            # 학습 관련 정보 갱신
            # epoch를 수행하는 동안 최대 포트폴리오 가치를 갱신하고
            # 해당 epoch에서 포트폴리오 가치가 자본금보다 높으면 epoch_win_cnt를 증가시킨다.
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        # 모든 epoch를 수행하고 전체 epoch 수행 소요 시간 기록
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        # with는 에러가 나든 안 나든 마지막에 무조건 close() 해주는 녀석
        with self.lock:
            # 주식 종목, 전체 소요 시간, 최대 포트폴리오 가치, 포트폴리오 가치가 자본금보다 높았던 epoch 수를 로그로 남긴다.
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} '
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')

    # save_models() : 학습을 마친 가치 신경망 및 정책 신경망 저장 함수
    # 신경망 안의 save_model() 함수를 호출해 파일로 저장한다.
    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)

    #
    def predict(self):
        # 에이전트 초기화
        self.agent.reset()

        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break

            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue

            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample))
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample))
            
            # 신경망에 의한 행동 결정
            action, confidence, _ = self.agent.decide_action(pred_value, pred_policy, 0)
            result.append((self.environment.observation[0], int(action), float(confidence)))
        # with는 에러가 나든 안 나든 마지막에 무조건 close() 해주는 녀석
        with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
            print(json.dumps(result), file=f)
        return result



# DQNLearner는 가치 신경망으로만 강화학습을 하는 방식이다.
class DQNLearner(ReinforcementLearner):
    # DQNLearner 생성자
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()   # 가치 신경망 생성을 위해 init_value_network() 함수 호출
    # 배치 학습 데이터 생성 함수
    # DQNLearner는 ReinforcementLearner 클래스를 상속하므로 ReinforcementLearner의 속성과 함수를 모두 가진다.
    # ReinforcementLearner를 상속하는 모든 클래스는 ReinforcementLearner의 추상 메소드인 get_batch() 함수를 꼭! 구현해야 한다.
    def get_batch(self):
        memory = zip(
            # memory_ 배열들을 역으로 묶어준다.
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        # 배열은 모두 0으로 채운다.
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))  # x: 샘플 배열
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))       # y_value: 레이블 배열
        value_max_next = 0
        # for문으로 샘플 배열과 레이블 배열에 값을 채운다.
        # memory를 역으로 취했기 때문에 for문은 배치 학습 데이터의 마지막 부분부터 처리한다.

        for i, (sample, action, value, reward) in enumerate(memory):
            x[i] = sample                           # x[i]에 샘플을 채워준다.
            r = self.memory_reward[-1] - reward     # 변수 r에 학습에 사용할 보상을 구해 저장한다.
            y_value[i] = value                      # y_value[i]에 가치 신경망의 출력을 넣어준다.
            y_value[i, action] = r + self.discount_factor * value_max_next  # 최대 가치에 할인율을 적용한 r을 구한다.
            value_max_next = value.max()                                    # 다음 상태의 최대 가치를 value_max_next 변수에 저장한다.
        return x, y_value, None
        # get_batch() 함수는 최종적으로 샘플 배열, 가치 신경망 학습 레이블 배열, 정책 신경망 학습 레이블 배열을 반환한다.
        # DQNLearner의 경우 '정책 신경망'을 사용하지 않기 때문에 정책 신경망 학습 레이블 배열 부분은 None으로 처리한다.


# PolicyGradientLearner는 정책 신경망으로만 강화학습을 하는 방식이다.
class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()  # '정책 신경망' 생성을 위해 init_policy_network() 함수 호출

    # 배치 학습 데이터 생성 함수
    # PolicyGradientLearner는 ReinforcementLearner 클래스를 상속하므로 ReinforcementLearner의 속성과 함수를 모두 가진다.
    # ReinforcementLearner를 상속하는 모든 클래스는 ReinforcementLearner의 추상 메소드인 get_batch() 함수를 꼭! 구현해야 한다.
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward         # 보상
            y_policy[i, :] = policy                     # y_policy[i]에 정책 신경망의 출력을 넣어준다.
            y_policy[i, action] = utils.sigmoid(r)      # sigmoid 함수를 취해서 정책 신경망 학습 레이블로 정한다.
        return x, None, y_policy
        # get_batch() 함수는 최종적으로 샘플 배열, 가치 신경망 학습 레이블 배열, 정책 신경망 학습 레이블 배열을 반환한다.
        # PolicyGradientLearner의 경우 '가치 신경망'을 사용하지 않기 때문에 가치 신경망 학습 레이블 배열 부분은 None으로 처리한다.


# ActorCriticLearner는 '가치 신경망', '정책 신경망' 모두를 사용하는 강화학습 방식이다.
class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None, 
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps, 
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

    # 배치 학습 데이터 생성 함수
    # ActorCriticLearner는 ReinforcementLearner 클래스를 상속하므로 ReinforcementLearner의 속성과 함수를 모두 가진다.
    # ReinforcementLearner를 상속하는 모든 클래스는 ReinforcementLearner의 추상 메소드인 get_batch() 함수를 꼭! 구현해야 한다.
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        # 위의 DQN, PolicyGradient와 다르게 y_value, y_policy 둘 다 가진다.
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next  # DQN과 동일하게 레이블을 넣어준다.
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(r)          # sigmoid를 취했기 때문에 정책 신경망 학습 label은
                                                            # 예측 가치가 양수면 0.5 이상이 되고 음수면 0.5 미만이 된다. 0~1 사이의 값을 가진다
            value_max_next = value.max()
        return x, y_value, y_policy
        # get_batch() 함수는 최종적으로 샘플 배열, 가치 신경망 학습 레이블 배열, 정책 신경망 학습 레이블 배열을 반환한다.

# ActorCriticLearner와 거의 유사하다.
# 다만 정책 신경망을 학습할 때 가치 신경망의 값을 그대로 사용하지 않고 Advantage를 사용한다.
# A2CLearner는 ActorCriticLearner 클래스를 상속한다.
class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # get_batch() 함수에서는 Advantage로 정책 신경망을 학습한다.
    # Advantage는 어떠한 상태에서 어떠한 행동이 다른 행동보다 얼마나 더 가치가 높은지를 의미한다.
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = reward_next + self.memory_reward[-1] - reward * 2
            reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()  # advantage는 상태,행동 가치에서 상태 가치를 뺀 값이다.
            y_policy[i, :] = policy
            y_policy[i, action] = utils.sigmoid(advantage)      # advantage를 sigmoid 함수에 적용해 정첵 신경망의 학습 레이블로 적용한다.
            value_max_next = value.max()
        return x, y_value, y_policy

# A3C는 A2C를 병렬로 수행하는 강화학습이다.
# A3C 역시 가치 신경망과 정책 신경망을 모두 사용한다.
class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None, 
        list_chart_data=None, list_training_data=None,
        list_min_trading_price=None, list_max_trading_price=None, 
        value_network_path=None, policy_network_path=None,
        **kwargs):
        # assert는 뒤에 조건문이 True가 아니면 Error를 발생시킨다.
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps, 
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data, 
            min_trading_price, max_trading_price) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_price, list_max_trading_price
            ):
            learner = A2CLearner(*args, 
                stock_code=stock_code, chart_data=chart_data, 
                training_data=training_data,
                min_trading_price=min_trading_price, 
                max_trading_price=max_trading_price, 
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    # A3CLearner 클래스의 run() 함수에서는 스레드를 이용해 각 A2CLearner 클래스 객체의 run() 함수를 동시에 수행한다.
    def run(self, learning=True):
        threads = []
        for learner in self.learners:
            # 파이썬에서 threading 모듈의 Thread 클래스를 사용한다.
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={'learning': learning}
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def predict(self):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.predict, daemon=True
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
