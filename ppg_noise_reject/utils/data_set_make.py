import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from template_gen import temp_find
from ppg_noise_reject.utils.label_template import viewer

class data_set_make:
    def __init__(self, link = None):
        # if ~os.chdir(link):
        #     print("The dataset link is nit found")
        self.link = link

    def single(self, file = None, visualize = None):
        peaks = None
        temp = None
        if file != None:
            temp = None
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                temper = temp_find()
                temper.ppg = data["PPG"]
                peaks,temp = temper.temping()

            if isinstance(visualize,list) and temp != None:
                rows = visualize[0]
                cols = visualize[1]
                y_peaks = [temper.ppg[i] for i in peaks]
                plt.plot(temper.ppg)
                plt.plot(peaks, y_peaks, "o")
                plt.show()
                fig, axes = plt.subplots(rows, cols, figsize=(visualize[2],visualize[3]))

                for i,sig in enumerate(temp):
                    r = i//cols
                    c = i%cols
                    if r < rows and c < cols:
                        axes[r, c].plot(sig)
                    else:
                        break

                plt.tight_layout()
                plt.show()
        return peaks, temp
    
    def gui_make(self):
        pass

#Test code 
if __name__ == "__main__":
    tester = data_set_make("test.json")
    tester.single("D:/ppg_project/code/model_build/my_tool/ppg_noise_reject/utils/test.json",[5,7,10,6])          