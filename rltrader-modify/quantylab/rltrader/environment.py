
#
class Environment:
    PRICE_IDX = 4  # 종가의 위치가 5번째 column이다.



    def __init__(self, chart_data=None):
        self.chart_data = chart_data    # 주식 종목의 차트 데이터 (2차원 배열 => DataFrame)
        self.observation = None         # 현재 관측치
        self.idx = -1                   # 차트 데이터에서의 현재 위치

    # idx와 observation을 초기화 => 차트 데이터의 처음으로 돌아가게 한다.
    def reset(self):
        self.observation = None
        self.idx = -1

    # idx를 다음 위치로 이동하고 observation을 업데이트
    def observe(self):
        if len(self.chart_data) > self.idx + 1: # 전체 데이터의 길이보다 다음 위치가 작을 경우 데이터를 가져온다.
            self.idx += 1                       # 하루 앞으로 이동
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None                             # 더 이상 제공할 데이터 없을 때 None return

    # 현재 observation에서 종가를 획득
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX] # 관측 데이터로부터 종가를 가져와 return
        return None
