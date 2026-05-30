class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder,mode = "Test"):

        self.signals = []
        self.labels = []
        self.target_length = 128
        self.a = temp_find()

        json_files = [f for f in os.listdir(root_folder) if f.endswith(".json")]
        print(f"Found {len(json_files)} json files.")

        for fname in json_files:
          # try:
            path = os.path.join(root_folder, fname)

            with open(path, "r") as file:
                data = json.load(file)

            s = data.get("PPG",[])
            self.a.ppg = data["PPG"]
            no_dc = self.a.dc_take(self.a.ppg)
            peaks,feet,temp = self.a.temping()
            spline = self.a.spline(feet, no_dc)
            final = no_dc - spline

            for ind,item in enumerate(data["POS"]):
              temp = np.array(final[item[0]:item[1]])
              temp = self.a.liesample(temp, self.target_length) # Use self.target_length
              temp = savgol_filter(temp, window_length=8, polyorder=3)
              min_val = temp.min()
              max_val = temp.max()

              # Handle cases where max_val - min_val is zero to prevent division by zero
              if (max_val - min_val) == 0:
                  # If all values are the same, normalize to all zeros or skip
                  print(f"Skipping segment from file {fname} due to zero range (all values are the same).")
                  continue
              else:
                  final_signal = (temp - min_val) / (max_val - min_val + 1e-8)

              self.signals.append(final_signal)
              label = [0, 0]
              if data["VALID"][ind] == 1:
                label[0] = 1
              else:
                label[0] = 0

              pos = item
              p1, p2 = pos
              label[1] = np.max(data["LABEL"][p1:p2])
              self.labels.append(label)

          # except Exception as e:
          #   print(f"Error processing file {fname}: {e}")
          #   pass


    def __getitem__(self, index):
        signal = self.signals[index]
        signal = torch.tensor(signal, dtype=torch.float32)
        signal = signal.unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return signal, label

    def __len__(self):
        return len(self.signals)