from scipy.sparse import hstack
from typing import List
from sklearn.feature_extraction.text import CountVectorizer


class TextFeatures:
    def __init__(self, category_en: List[str], shop_name_en: List[str]):
        self._catg_en = category_en
        self._shop_n_en = shop_name_en

        self._catg_encoder = None
        self._shop_n_encoder = None
        self._catg_stop_words = ['in', 'and', 'for', 'of', 'the']
        self._shop_n_stop_words = None

        self._catg_features = None
        self._shop_n_features = None

        self.fit()

    def _fit_catg(self):
        c = CountVectorizer(stop_words=self._catg_stop_words)
        self._catg_features = c.fit_transform(self._catg_en)

    def _fit_shop_n(self):
        c = CountVectorizer(stop_words=self._shop_n_stop_words)
        self._shop_n_features = c.fit_transform(self._shop_n_en)

    def fit(self):
        self._fit_catg()
        self._fit_shop_n()

    def one_hot_shop_name_features(self, shop_id: int):
        return self._shop_n_features[shop_id]

    def one_hot_category_name_features(self, category_id: int):
        return self._catg_features[category_id]

    def one_hot_features(self, shop_id: int, category_id: int):
        catg_f = self.one_hot_category_name_features(category_id)
        shop_nf = self.one_hot_shop_name_features(shop_id)
        return hstack([catg_f, shop_nf])


if __name__ == '__main__':
    dir = '/home/ashesh/Documents/initiatives/kaggle_competitions/sales_prediction/'
    category_en = open(dir + 'translated_categories.txt').read().splitlines()
    shop_name_en = open(dir + 'translated_shops.txt').read().splitlines()
    features = TextFeatures(category_en, shop_name_en)
    print('Category name features', features.one_hot_category_name_features(0).shape)
    print('Shop name features', features.one_hot_shop_name_features(1).shape)

    print('Net features', features.one_hot_features(0, 1).shape)
    print('Net featurs type', type(features.one_hot_features(0, 1)))
