import os
import sys
import logging
import argparse
import json

# 각자 본인의 local 환경에 맞게 설정
os.environ['RLTRADER_BASE'] = 'C:\\big14\\BKST\\rltrader-master'
from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader import data_manager

# 아나콘다 프롬프트에서 conda activate bkst 하고 main.py 가 있는 폴더까지 이동해서 실행
# ex) "C:\big14\BKST\rltrader-master
# Anaconda prompt에서 실행하는 방법 : python main.py --stock_code 005930
# 값들을 바꾸고 싶으면 추가해서 사용하면 된다. 책 239pg 참고

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v3')                            # RLTrader의 버전
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--stock_code', nargs='+', default=["005930"])                                      # 강화학습의 환경이 될 주식 종목 코드
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'], default='dqn')  # 강화학습 방식
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')                   # 가치 신경망과 정책 신경망 중 사용할 신경망 유형 선택
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')       # 백엔드로 사용할 Framework 설정
    parser.add_argument('--start_date', default='20200101')
    parser.add_argument('--end_date', default='20201231')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--discount_factor', type=float, default=0.7)                                       # 할인율
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    # JSON 형태로 저장
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    print('main: ', args)
    # args.stock_code의 class는 'str'
    for stock_code in args.stock_code:
        print('stock_code: ', stock_code)
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 100000
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {
              'rl_method': args.rl_method                   # 강화학습 방법
            , 'net': args.net                               # 신경망 종류
            , 'num_steps': num_steps                        # LSTM과 CNN에서 사용할 step 수
            , 'lr': args.lr                                 # 학습 속도
            , 'balance': args.balance                       # 현재 현금 잔고
            , 'num_epoches': num_epoches                    # epoch 수
            , 'discount_factor': args.discount_factor       # 할인율
            , 'start_epsilon': start_epsilon                # 시작 epsilon 값
            , 'output_path': output_path                    # 출력 경로
            , 'reuse_models': reuse_models}                 # 신경망 모델 재사용 여부

        # 강화학습 시작
        learner = None
        # 강화학습 종류에 맞게 강화학습 학습기 클래스를 정하고 가치 신경망과 정책 신경망의 경로를 지정한다.
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params,                # 학습기 class의 object를 생성한다.
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params,     # 학습기 class의 object를 생성한다.
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params,        # 학습기 class의 object를 생성한다.
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params,                # 학습기 class의 object를 생성한다.
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{                                        # 학습기 class의 object를 생성한다.
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)              # learner object의 run() 함수를 호출해 강화학습을 시작한다.
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
