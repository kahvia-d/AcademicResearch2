import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_YiChang_with_classes():
    # 获取当前文件（dataLoad.py）所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 项目根目录（假设utils在根目录下）
    project_dir = os.path.dirname(current_dir)
    # 拼接数据文件的绝对路径
    data_file = os.path.join(project_dir, 'data', 'input', '240329含等级数据.xlsx')
    selected_columns = [
        'ZH_CLASS',
        # 'DLBM',
        # 'Aspect',
        'DEM',
        'SOILCODE',
        'NDVI',
        # 'DIST_ming',
        'DIST_water',
        'DIST_trans',
        # 'DIST_human',
        'precipitation',
        'temperature',
        'pH',
        # 'Slope'
    ]
    df = pd.read_excel(data_file, usecols=selected_columns)

    columns_to_normalize = [col for col in selected_columns if col not in ['DLBM', 'SOILCODE', 'ZH_CLASS']]
    data_to_normalize = df[columns_to_normalize]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data_to_normalize)
    normalized_data = 2 * normalized_data - 1

    normalized_data = pd.DataFrame(normalized_data, columns=columns_to_normalize).round(2)
    # df['DLBM'] = pd.factorize(df['DLBM'])[0]
    df['SOILCODE'] = pd.factorize(df['SOILCODE'])[0]
    df_normalized = pd.concat([df[['ZH_CLASS','SOILCODE']], normalized_data], axis=1)

    # df_normalized = pd.concat([df[['ZH_CLASS', 'DLBM','SOILCODE']], normalized_data], axis=1)

    df_normalized = df_normalized[selected_columns]

    return df_normalized
