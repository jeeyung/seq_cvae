import array
import pickle
import gzip

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

if __name__ == "__main__":
    ImgF = readImageFeatures('./data/amazon/image_features_Clothing_Shoes_and_Jewelry.b')
    with open('./data/amazon_min10_woman/item_list.pkl', 'rb') as i:
        item_list = pickle.load(i)
    with open('./data/amazon_min10_woman/show2id.pkl', 'rb') as i:
        show2id = pickle.load(i)
    # no_train_item=[]
    while True:
        try:
            f_name, features = next(ImgF)
        except:
            break
        else:
            fn = f_name.decode("utf-8")
            if fn in item_list:
                try:
                    sid = show2id[fn]
                except:
                    # no_train_item.append(fn)
                    # print(no_train_item)
                    continue
                else:
                    with open(f'./data/amazon_min10_woman/image_cnn/{sid}.pkl', 'wb') as f:
                        pickle.dump(features, f)
                    print(f'{fn} saved as {sid}.pkl')
            else:
                continue

# if __name__ == "__main__":
#     meta = parse('./data/amazon/meta_Clothing_Shoes_and_Jewelry.json.gz')
#     # with open('./data/amazon_min20/item_list.pkl', 'rb') as f:
#         # item_list = pickle.load(f)
#     woman_img = []
#     man_id = []
#     i=0
#     while i<10000:
#         try:
#             data = next(meta)
#             # if data['asin'] in item_list:
#             for categories in data['categories']:
#                 print(categories)
#                 if set(categories).intersection(set(['Clothing'])):
#                     if data['asin'] not in woman_img:
#                         woman_img.append(data['imUrl'])
#             i += 1
#         except:
#             continue

#     print('valid', woman_img)
                    # if set(categories).intersection(set(['Men'])):
                        # if data['asin'] not in man_id:
                            # man_id.append(data['asin'])
            # a -= 1
            # print(len(woman_id), len(man_id))
        # except:
            # with open('./data/amazon_min20/woman_id.pkl', 'wb') as f:
                # pickle.dump(woman_id, f)
            # with open('./data/amazon_min20/man_id.pkl', 'wb') as f:
                # pickle.dump(man_id, f)
            # break
    # print('woman', woman_id, 'man', man_id)

# if __name__ == "__main__":
#     meta = parse('./data/amazon_min20/meta_Clothing_Shoes_and_Jewelry.json.gz')
#     with open('./data/amazon_min10_woman/item_list.pkl', 'rb') as f:
#         item_list = pickle.load(f)
#     woman_id = []
#     man_id = []
#     while True:
#         try:
#             data = next(meta)
#             if data['asin'] in item_list:
#                 for categories in data['categories']:
#                     if set(categories).intersection(set(['Women'])):
#                         if data['asin'] not in woman_id:
#                             woman_id.append(data['asin'])
#                     if set(categories).intersection(set(['Men'])):
#                         if data['asin'] not in man_id:
#                             man_id.append(data['asin'])
#             # a -= 1
#             print(len(woman_id), len(man_id))
#         except:
#             with open('./data/amazon_min10_woman/woman_id.pkl', 'wb') as f:
#                 pickle.dump(woman_id, f)
#             # with open('./data/amazon_min10/man_id.pkl', 'wb') as f:
#                 # pickle.dump(man_id, f)
#             break
#     print('woman', woman_id, 'man', man_id)
