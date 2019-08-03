import array
import pickle

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

if __name__ == "__main__":
    ImgF = readImageFeatures('./data/amazon/image_features_Clothing_Shoes_and_Jewelry.b')
    with open('./data/amazon/item_list.pkl', 'rb') as i:
        item_list = pickle.load(i)
    with open('./data/amazon/show2id.pkl', 'rb') as i:
        show2id = pickle.load(i)
    while True:
        try:
            f_name, features = next(ImgF)
        except:
            break
        else:
            fn = f_name.decode("utf-8")
            if fn in item_list:
                sid = show2id[fn]
                with open(f'./data/amazon/image_cnn/{sid}.pkl', 'wb') as f:
                    pickle.dump(features, f)
                print(f'{fn} saved as {sid}.pkl')
            else:
                continue