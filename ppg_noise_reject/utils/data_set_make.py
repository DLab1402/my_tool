import os
import json
import numpy as np
import matplotlib.pyplot as plt

from template_gen import temp_find

class data_set_make:
    def __init__(self, link = None):
        if ~os.chdir(link):
            print("The dataset link is nit found")
        self.link = link

    def single(self, file = None, visualize = None):
        if file != None:
            temp = None
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                temper = temp_find()
                temper.ppg = data["PPG"]
                temp = temper.temping()

            if isinstance(visualize,list) and temp != None:
                try:
                    rows = visualize[0]
                    cols = visualize[1]

                    fig, axes = plt.subplots(rows, cols, figsize=(visualize[2],visualize[3]))

                    for i,sig in enumerate(temp):
                        r = i//cols
                        c = i%cols
                        if r <= rows:
                            axes[r, c].plot(sig)

                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(e)
                    

#Test code 
if __name__ == "__main__":
    tester = data_set_make("D:\\ppg_project\\code\\model_build\\my_tool\\test\\0001331_14076356_wave.csv.json")
    tester.single("D:\\ppg_project\\code\\model_build\\my_tool\\test\\0001331_14076356_wave.csv.json",[2,3,10,6])          