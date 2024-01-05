import pymysql

class DAO:
    def __init__(self,host="localhost",user="root",password="ai305312",database="douyin_analyzer") -> None:
        '''
        初始化数据库
        '''
        self.db = pymysql.connect(host="localhost",user="root",password="ai305312",database="douyin_analyzer")
    
    def get_通用敏感词(self):
        '''
        从数据库中获取通用敏感词列表
        '''
        sql = """
        select * from 通用敏感词;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_通用敏感词id(self, word:str):
        '''
        查询敏感词的id
        '''
        sql = """
        select id from 通用敏感词 where 敏感词 = '{word}';
        """.format(word=word)
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def insert_专项变体词(self,原词:str,变体词:str):
        '''
        向数据库中插入专项变体词
        '''
        sql = '''
        INSERT INTO 专项变体词 (原词, 变体词)
        VALUES ('{原词}', '{变体词}');
        '''.format(原词=原词,变体词=变体词)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()

    def get_专项变体词(self):
        '''
        从数据库中查询专项变体词
        '''
        sql = '''
        select * from 专项变体词匹配;
        '''
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_专项变体词id(self, word:str):
        '''
        查询专项变体词的id
        '''
        sql = """
        select id from 专项变体词 where 变体词 = '{word}';
        """.format(word=word)
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_保健品(self):
        '''
        从数据库中获取保健品列表
        '''
        sql = """
        select * from 保健品;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def get_国产化妆品(self):
        '''
        从数据库中获取国产化妆品列表
        '''
        sql = """
        select * from 国产化妆品;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_进口化妆品(self):
        '''
        从数据库中获取进口化妆品列表
        '''
        sql = """
        select * from 进口化妆品;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_国产药品(self):
        '''
        从数据库中获取国产药品列表
        '''
        sql = """
        select * from 国产药品;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_进口药品(self):
        '''
        从数据库中获取进口药品列表
        '''
        sql = """
        select * from 进口药品;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_禁售限售(self):
        '''
        从数据库中获取禁售限售物品列表
        '''
        sql = """
        select * from 禁售限售;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_专项变体词(self):
        '''
        从数据库中获取专项变体词列表
        '''
        sql = """
        select * from 专项变体词;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def get_进口医疗器械备案(self):
        '''
        从数据库中获取进口医疗器械(备案)列表
        '''
        sql = """
        select * from 进口医疗器械备案;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_进口医疗器械注册(self):
        '''
        从数据库中获取进口医疗器械(注册)列表
        '''
        sql = """
        select * from 进口医疗器械注册;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_国产医疗器械备案(self):
        '''
        从数据库中获取国产医疗器械(备案)列表
        '''
        sql = """
        select * from 国产医疗器械备案;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_国产医疗器械注册(self):
        '''
        从数据库中获取国产医疗器械(注册)列表
        '''
        sql = """
        select * from 国产医疗器械注册;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_产品匹配(self):
        '''
        从数据库中获取产品匹配记录表
        '''
        sql = """
        select * from 产品匹配;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def insert_产品匹配(self,live_id:str,product_class:str,product_id:str,conflict:str):
        '''
        向数据库中插入产品匹配记录表
        '''
        sql = """
        INSERT INTO 产品匹配 (live_id, product_class, product_id, conflict)
        VALUES ('{live_id}', '{product_class}', '{product_id}', '{conflict}');
        """.format(live_id=live_id, product_class=product_class, product_id=product_id, conflict=conflict)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()
    
    def get_禁售限售匹配(self):
        '''
        从数据库中获取禁售限售匹配记录表
        '''
        sql = """
        select * from 禁售限售匹配;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def insert_禁售限售匹配(self,live_id:str,禁售限售_id:str):
        '''
        向数据库中插入禁售限售匹配记录
        '''
        sql = """
        INSERT INTO 禁售限售匹配 (live_id, 禁售限售_id)
        VALUES ('{live_id}', '{禁售限售_id}');
        """.format(live_id=live_id, 禁售限售_id=禁售限售_id)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()
    

    
    def get_通用敏感词匹配(self):
        '''
        从数据库中获取通用敏感词匹配记录表
        '''
        sql = """
        select * from 通用敏感词匹配;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def insert_通用敏感词匹配(self,live_id:str,通用敏感词_id:str):
        '''
        向数据库中插入通用敏感词匹配记录
        '''
        sql = """
        INSERT INTO 通用敏感词匹配 (live_id, 通用敏感词_id)
        VALUES ('{live_id}', '{通用敏感词_id}');
        """.format(live_id=live_id, 通用敏感词_id=通用敏感词_id)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()

    def get_专项变体词匹配(self):
        '''
        从数据库中获取专项变体词匹配记录表
        '''
        sql = """
        select * from 专项变体词匹配;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def insert_专项变体词匹配(self,live_id:str,专项变体词_id:str):
        '''
        向数据库中插入专项变体词匹配记录
        '''
        sql = """
        INSERT INTO 专项变体词匹配 (live_id, 专项变体词_id)
        VALUES ('{live_id}', '{专项变体词_id}');
        """.format(live_id=live_id, 专项变体词_id=专项变体词_id)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()

    def get_live(self):
        '''
        从数据库中获取live记录表
        '''
        sql = """
        select * from live;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_live_id_max(self):
        '''
        从数据库中获取live id 最大值, 即当前最新的id
        '''
        sql = """
        select max(id) from live;
        """
        cursor = self.db.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        return result

    def insert_live(self, origin_url:str, live_name:str, video_path:str, ocr_path:str, asr_path:str):
        '''
        向数据库中插入live记录
        '''
        sql = """
        INSERT INTO live (origin_url, live_name, video_path, ocr_path, asr_path)
        VALUES ('{origin_url}', '{live_name}', '{video_path}', '{ocr_path}', '{asr_path}');
        """.format(origin_url=origin_url, live_name=live_name, video_path=video_path, ocr_path=ocr_path, asr_path=asr_path)
        cursor = self.db.cursor()
        cursor.execute(sql)
        self.db.commit()
        cursor.close()
    
if __name__ == "__main__":
    # test_测试()
    dao = DAO()
    # dao.insert_产品匹配("123","adsf","wqq","afd")
    # dao.insert_禁售限售匹配("123","123")
    # dao.insert_通用敏感词匹配("123","123")
    # dao.insert_live("testurl",'testurl','testname','testpath','testpath','testpath')
    # words = dao.get_通用敏感词()
    # words = [i[1] for i in words]
    # print(words)
    print(type(dao.get_live_id_max()[0][0]))
    print(dao.get_通用敏感词id("小脑")[0][0])
    # dao.insert_通用敏感词匹配(str(0),str(0))