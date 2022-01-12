import os
import pandas
import shutil

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__=='__main__':
    dump = []
    for i in range(1, 5):
        filepath = os.path.join(BASE_DIR, 'preprocess', f'train_meta_{i}.csv')
        print(filepath)
        df = pandas.read_csv(filepath)
        dump.append(df)
    dump = pandas.concat(dump)

    for index,row in dump.iterrows():
        shutil.copy(os.path.join(BASE_DIR, 'train', 'images', row['fname']), os.path.join(BASE_DIR, 'my_images', row['fname']))

    print('Done !')
        
        
        
        
    
    
    