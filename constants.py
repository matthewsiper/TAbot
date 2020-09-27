CBUILDER_WORKER_QUEUES = ["cb_queue1", "cb_queue2", "cb_queue3"]

FINVIZ_URLS = {"cb_queue1": [{"ta_pattern_horrizontal2": ["AEP", "ARCM", "ATHM", "CBMB", "DXF", "EVBG", "FGBI", "FUMB", "FVCB"]}],
               "cb_queue2": [{"ta_pattern_wedgeup2": ["BNSO", "BSJR", "CACC", "CARZ", "CE", "CHCT", "CLNE", "CMO", "CNXN"]}],
               "cb_queue3": [{"ta_pattern_wedgedown2": ["BROG", "CPTA", "ESBA", "ESRT", "HII", "SALT", "HII", "HII",
                                                        "HII"]}]}



TICKERS = {"queue1": ["TSLA"],
           "queue2": ["AAPL"],
           "queue3": ["MSFT"]}

BASE_MODELS_DIR = "/Users/mattscomputer/6513 P2/"
PRED_CHART_DIR = "/Users/mattscomputer/6513 P2/live"

TICKERS_FOR_DROPDOWN = [{"label": "QQQ", "value": "QQQ"},
                        {"label": "AAPL", "value": "AAPL"},
                        {"label": "TSLA", "value": "TSLA"},
                        {"label": "MSFT", "value": "MSFT"},
                        {"label": "AMZN", "value": "AMZN"},
                        {"label": "GS", "value": "GS"},
                        {"label": "DKNG", "value": "DKNG"},
                        {"label": "T", "value": "T"},
                        {"label": "BROG", "value": "BROG"},
                        {"label": "CPTA", "value": "CPTA"}]

PERIODICITY = "120d"
CHART_PATTERN = "live"

PRED_DICT = {0: "Horrizontal", 1: "Wedgedown", 2: "Wedgeup"}



P_CACHE = {'QQQ': {'model1': (1.0, 'Wedgeup'),
                   'model2': (0.99594745, 'Wedgeup'),
                   'model3': (0.77594745, 'Wedgeup')},
           'AAPL': {'model1': (1.0, 'Horrizontal'),
                    'model2': (0.9998878, 'Horrizontal'),
                    'model3': (0.71205276, 'Wedgeup')},
           'TSLA': {'model1': (1.0, 'Horrizontal'),
                    'model2': (0.8999651, 'Horrizontal'),
                    'model3': (0.7617307, 'Wedgeup')},
           'BROG': {'model1': (1.0, 'Wedgedown'),
                    'model2': (0.99882704, 'Wedgedown'),
                    'model3': (0.736952, 'Wedgeup')},
           'MSFT': {'model1': (1.0, 'Horrizontal'),
                    'model2': (0.99992347, 'Horrizontal'),
                    'model3': (0.71943575, 'Wedgeup')},
           'AMZN': {'model1': (1.0, 'Horrizontal'),
                    'model2': (0.99992704, 'Horrizontal'),
                    'model3': (0.736952, 'Wedgeup')},
           'T': {'model1': (1.0, 'Wedgedown'),
                 'model2': (0.99999964, 'Wedgedown'),
                 'model3': (0.61114573, 'Wedgeup')},
           'GS': {'model1': (1.0, 'Horrizontal'),
                  'model2': (0.99997306, 'Horrizontal'),
                  'model3': (0.79304683, 'Wedgeup')},
           'DKNG': {'model1': (1.0, 'Horrizontal'),
                    'model2': (0.9999033, 'Horrizontal'),
                    'model3': (0.7996113, 'Wedgeup')},
           'CPTA': {'model1': (1.0, 'Wedgedown'),
                    'model2': (0.99992347, 'Wedgedown'),
                    'model3': (0.71205276, 'Horrizontal')},

           }
