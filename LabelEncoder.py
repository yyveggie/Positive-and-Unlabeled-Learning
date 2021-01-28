'''将需要量化的数据进行量化'''

from Conversion import Quantify

path = r'C:\Users\yyveggie\Desktop\UCI\mushroom.data'
texts = Quantify(path)
save_path = r'C:\Users\yyveggie\Desktop\UCI\Conversion\mushroom.csv'
texts.to_csv(save_path, header=False, index=False)