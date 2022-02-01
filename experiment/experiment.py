import json
import os
from datetime import datetime

class Experiment:
    ''' experiment details '''
    def __init__(self, name, output_dir='results/'):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.name = f'{name}_{current_time}'
        self.out_path = os.path.join(output_dir,  self.name)
        os.makedirs(self.out_path, exist_ok=True)

        self.results = {}

    # def check_last_version(dirname):
    #     v = 0 
    #     while os.path.exists(dirname + str(v)):
    #         v += 1
    #     return v
  
    def store_json(self):
        with open(os.path.join(self.out_path, "JSONDump.txt"), 'w') as outfile:
            json.dump(json.dumps(self.__dict__), outfile)

if __name__ == "__main__":
    e = Experiment("TestExperiment")
    e.results["Temp Results"] = [[1, 2, 3, 4], [5, 6, 2, 6]]
    e.results["Temp Results2"] = ["dfgdfg"]
    e.store_json()
