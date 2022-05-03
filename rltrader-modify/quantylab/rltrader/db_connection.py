import pymysql

db = pymysql.connect(host='34.64.33.94', user='test', db='test', password='test', charset='utf8')

cursor = db.cursor()


def insert(info_relate_epoch):
    # 파이썬 문자열 형식화를 사용해서 dict에 있는 key를 사용해서
    # 직관적으로 어떤 값을 썼는지 알 수 있다.

    # info_relate_epoch에 있는 values()를 tuple로 변환해서 사용
    # 근데 values에 뭐가 있는지 바로 알기 힘들다.

    sql = """ 
            insert into log_table_test(
                  stock_code          
                , epoch               
                , epsilon             
                , expl                
                , buy                 
                , sell                
                , hold                
                , stocks              
                , PV                  
                , loss                
                , ET  
            )          
            values(    %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s      
                     , %s
            )          
    """
    cursor.execute(sql, tuple(info_relate_epoch.values()))
    db.commit()  # 커밋 시 데이터베이스 반영

