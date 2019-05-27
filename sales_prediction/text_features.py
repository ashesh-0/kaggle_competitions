import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize


def _get_text_features(names, num_features):
    tfidf = TfidfVectorizer()
    features = tfidf.fit_transform(names).toarray()
    print('Num features from tfidf', features.shape[1])
    if features.shape[1] <= num_features or features.shape[0] <= num_features:
        return features.astype(np.float32)

    print('Using NMF')
    pca = NMF(n_components=num_features)
    return pca.fit_transform(normalize(features)).astype(np.float32)


def _get_text_feature_df(df, id_col, text_col, num_features):
    features = _get_text_features(df[text_col].tolist(), num_features)
    cols = ['{}_text_{}'.format(text_col, i) for i in range(features.shape[1])]
    output_df = pd.DataFrame(features, columns=cols).astype(np.float32)
    output_df[id_col] = df[id_col].tolist()
    return output_df


def get_item_text_features(items_df, num_features):
    return _get_text_feature_df(items_df, 'item_id', 'item_name', num_features)


def get_shop_text_features(shops_df, num_features):
    return _get_text_feature_df(shops_df, 'shop_id', 'shop_name', num_features)


def get_category_text_features(categories_df, num_features):
    return _get_text_feature_df(categories_df, 'item_category_id', 'item_category_name', num_features)
