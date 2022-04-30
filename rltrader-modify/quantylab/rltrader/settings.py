import os
import locale
import platform


# 로거 이름
LOGGER_NAME = 'rltrader'


# 경로 설정
# os.environ.get('key') : 특정 key에 대한 시스템 환경 변수를 가져옴
# os.path.abspath() : 절대 경로 구하기
# os.path.join() : 경로명 조작에 관한 처리(2개 이상의 경로를 결합하여 하나의 경로로 지정 가능)
BASE_DIR = os.environ.get('RLTRADER_BASE', 
    os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))


# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
