'''Quantify the attributes that need it'''

from Conversion import Quantify

path = r'data/mushroom.data'
texts = Quantify(path)
save_path = r'data/mushroom.csv'
texts.to_csv(save_path, header=False, index=False)