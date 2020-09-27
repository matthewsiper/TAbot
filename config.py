CONFIG_PARAMS = {
    "aws_pub_domain": "ec2-user@ec2-3-83-97-166.compute-1.amazonaws.com",
    "mongodb_params": {
        "host": "localhost", #change later when move to cluster
        "port": 27017
    },
    "chart_builder_params": {
        "chart_home_dir": "/home/ec2-user/6513 P2"
    },
    "td_ameritrade_api": {
        "base": "https://api.tdameritrade.com",
        "uri": "/v1/marketdata/{}/pricehistory?apikey=FAZ4GPL25XE2CA5X1HPOUBGLXHRC9VK6&periodType=day&period=1&frequencyType=minute&frequency=5"
    },
    "finviz_api": {
        "base": "https://finviz.com/screener.ashx?v=111&f={}&ft=3",
        "ta_patterns": ["ta_pattern_horizontal2", "ta_pattern_wedgedown2", "ta_pattern_wedgeup2"]
    }
}


class Config(object):
    def __init__(self, key_params):
        self.params = CONFIG_PARAMS[key_params]