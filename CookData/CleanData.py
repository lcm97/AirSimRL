import pandas as pd
from scipy.ndimage import gaussian_filter
import cv2
import numpy
import os
import DNNmodel.ImageComplexity
RAW_DATA_DIR = 'C:\dataset\AirSimData\data_raw\\320'

df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'airsim_rec.txt'), sep='\t')
print(df.head())
# df.drop(['POS_X', 'POS_Y', 'POS_Z', 'Q_W', 'Q_X', 'Q_Y', 'Q_Z'], axis = 1, inplace = True)
# print(df.head())
# df.insert(1,'Steering',0.)
# df.insert(2,'Collision',0)
# df.insert(3,'Complexity',0)
# print(df.head())

for i in range(len(df)):
    print(df.loc[i, 'Collision'])
    if df.loc[i, 'Collision'] == 0:
        df.loc[i, 'Collision'] = 0.0


# for i in range(len(df)):
#     print(df.loc[i, 'ImageFile'])
#     df.loc[i, 'ImageFile'] = str(i)+'.png'
#
# for i in range(len(df)):
#     print(df.loc[i, 'Collision'])
#     if i >= 832:
#         df.loc[i, 'Collision'] = 1
#
# for i in range(len(df)):
#     print(df.loc[i, 'ImageFile'])
#     image_name = 'images/' + df.loc[i, 'ImageFile']
#     img = cv2.imread(os.path.join(RAW_DATA_DIR, image_name))
#     result = gaussian_filter(img, sigma=1)
#     Sob = DNNmodel.ImageComplexity.Sobel(result)
#     img_sob = Sob.run()
#     df.loc[i, 'Complexity'] = numpy.mean(img_sob)
#     print(df.loc[i, 'ImageFile'] + " ====> " + str(numpy.mean(img_sob)))

# df.to_csv(os.path.join(RAW_DATA_DIR, 'airsim_rec_modified.txt'),sep='\t',float_format='%.6f')