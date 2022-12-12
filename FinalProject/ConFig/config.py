import json, os , numpy as np
absolute_path = os.getcwd()
absolute_path
model_path=os.path.join(absolute_path, 'data/data.csv')
print(model_path)


class ConfigReader():
    def __init__(self, conf_path= "media/SSD1TB/rezaei/Projects/GuidedCTCOCR/guidedctcocr/ConFig/config.json"):
        with open("/media/SSD1TB/rezaei/Projects/FinalProject/ConFig/config.json", "r") as f:
            cfg = json.load(f)
#    def __init__(self):
        # # with open('/media/SSD1TB/rezaei/Projects/FinalProject/data/data.csv') as f:#"E:/Fanap/ConFig/config.json", "r") as f:
        # #     cfg = json.load(f)
        # with open('/media/SSD1TB/rezaei/Projects/FinalProject/data/data.csv') as f:#"E:/Fanap/ConFig/config.json", 'r') as j:

             #cfg = json.loads(f.read())
        self.modelName = cfg["modelName"]
        self.modelType = cfg["modelType"]
        self.SanityCheck = cfg["SanityCheck"] == "True"
        self.batchSize  = np.int0(cfg["batchSize"])
        self.TotalEpoch = np.int0(cfg["TotalEpoch"])
        self.NumGRUlayer = np.int0(cfg["NumGRUlayer"])
        self.NumGRUunit = np.int0(cfg["NumGRUunit"])
        self.LSTMunit = np.int0(cfg["LSTMunit"])
        self.dropout = np.float32(cfg["dropout"])
        self.lr =np.float32(cfg["lr"])
        self.Shuffle = cfg["Shuffle"]
        self.MaxSeqLength = 2000
        self.MainPath = "/media/Archive4TB3/Data/textImages/EN_Benchmarks/"
        self.num_classes = 10
        self.valsplit =np.float32(cfg["valsplit"])
ConfigReader()