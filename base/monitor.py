import pandas as pd
import sys
class Monitor:

    def __init__(self):
        self.values = pd.DataFrame()
        self.finalized = False

    def finalize(self):
        self.values['loss_group'] = self.values['loss_group'].fillna(0.)
        self.values['epoch'] = self.values['epoch'].astype(int)

    def store(self, dict_values):
        self.values = self.values.append(dict_values, ignore_index = True)
        #self.values = pd.concat([self.values, pd.DataFrame(dict_values)], ignore_index=True)

    def get_values(self, key):
        self.finalize()
        return self.values.groupby('epoch')[key].mean()

    def get_epoch_values(self, key, epoch):
        return self.values[self.values['epoch']==epoch][key].mean()

    def get_epoch_values_SUM(self, key, epoch):
        return self.values[self.values['epoch']==epoch][key].sum()

    def save(self, path_name):
        self.finalize()
        self.values.to_csv(path_name, index=False)

