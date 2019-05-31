from geopy import distance
from geopy.geocoders import Nominatim

CITY_DATA = {
    'Калуга': {
        'type': 'city',
        'importance': 0.657295410002972,
        'area': 358.911060027607,
        'lat': 54.5101087,
        'lon': 36.2598115
    },
    'Волжский': {
        'type': 'city',
        'importance': 0.550220481385513,
        'area': 438.60566745306613,
        'lat': 48.782102,
        'lon': 44.7779843
    },
    'Воронеж': {
        'type': 'city',
        'importance': 0.7045722199183749,
        'area': 2087.0423692794884,
        'lat': 51.6605982,
        'lon': 39.2005858
    },
    'Жуковский': {
        'type': 'city',
        'importance': 0.5444576798578901,
        'area': 84.74916146073933,
        'lat': 55.5972801,
        'lon': 38.1199863
    },
    'Самара': {
        'type': 'city',
        'importance': 0.696449565028584,
        'area': 2205.1176282331953,
        'lat': 53.198627,
        'lon': 50.113987
    },
    'Коломна': {
        'type': 'city',
        'importance': 0.648802312001607,
        'area': 149.8134437525239,
        'lat': 55.0938743,
        'lon': 38.7670121
    },
    'Якутск': {
        'type': 'city',
        'importance': 0.654131338136001,
        'area': 304.64719883743965,
        'lat': 62.027287,
        'lon': 129.732086
    },
    'Ярославль': {
        'type': 'city',
        'importance': 0.7065300207968069,
        'area': 455.605256943589,
        'lat': 57.6263877,
        'lon': 39.8933705
    },
    'Балашиха': {
        'type': 'city',
        'importance': 0.604005896721309,
        'area': 341.3399072537641,
        'lat': 55.8036225,
        'lon': 37.9646488
    },
    '!Якутск': {
        'type': 'city',
        'importance': 0.654131338136001,
        'area': 304.64719883743965,
        'lat': 62.027287,
        'lon': 129.732086
    },
    'Новосибирск': {
        'type': 'city',
        'importance': 0.728185699386452,
        'area': 977.968425123674,
        'lat': 55.0282171,
        'lon': 82.9234509
    },
    'Тюмень': {
        'type': 'city',
        'importance': 0.672385913300304,
        'area': 1029.734170100062,
        'lat': 57.153534,
        'lon': 65.542274
    },
    'Москва': {
        'type': 'city',
        'importance': 0.922316290384526,
        'area': 2220.463036723328,
        'lat': 55.7504461,
        'lon': 37.6174943
    },
    'Томск': {
        'type': 'city',
        'importance': 0.680307551446071,
        'area': 378.01721837627406,
        'lat': 56.488712,
        'lon': 84.952324
    },
    'Казань': {
        'type': 'city',
        'importance': 0.741972620362528,
        'area': 1317.0415739017471,
        'lat': 55.7823547,
        'lon': 49.1242266
    },
    'Курск': {
        'type': 'city',
        'importance': 0.676179816540548,
        'area': 399.62069154089585,
        'lat': 51.739433,
        'lon': 36.179604
    },
    'Уфа': {
        'type': 'city',
        'importance': 0.695077392339603,
        'area': 1721.7276870129429,
        'lat': 54.726288,
        'lon': 55.947727
    },
    'Вологда': {
        'type': 'city',
        'importance': 0.661524874526942,
        'area': 214.3430466663635,
        'lat': 59.218876,
        'lon': 39.893276
    },
    'Ростов На Дону': {
        'type': 'city',
        'importance': 0.7893194127943861,
        'area': 811.0162296285571,
        'lat': 47.2213858,
        'lon': 39.7114196
    },
    'Адыгея': {
        'type': 'administrative',
        'importance': 0.692337665336592,
        'area': 27295.264468374502,
        'lat': 44.6939006,
        'lon': 40.1520421
    },
    'Химки': {
        'type': 'city',
        'importance': 0.614359398958108,
        'area': 230.77301628863356,
        'lat': 55.8892847,
        'lon': 37.4449896
    },
    'Чехов': {
        'type': 'town',
        'importance': 0.49824054756674396,
        'area': 38.796925301165096,
        'lat': 55.1426603,
        'lon': 37.4545328
    },
    'Мытищи': {
        'type': 'city',
        'importance': 0.617191804459829,
        'area': 95.08539485314338,
        'lat': 55.9094928,
        'lon': 37.7339358
    },
    'Красноярск': {
        'type': 'city',
        'importance': 0.698696821640559,
        'area': 836.7336288346606,
        'lat': 56.0090968,
        'lon': 92.8725147
    },
    'Сергиев': {
        'type': 'city',
        'importance': 0.529283352524288,
        'area': 88.864113189393,
        'lat': 56.3153529,
        'lon': 38.1358208
    },
    'Сургут': {
        'type': 'city',
        'importance': 0.617507029143067,
        'area': 315.54371943139705,
        'lat': 61.254032,
        'lon': 73.3964
    },
    'СПб': {
        'type': 'city',
        'importance': 0.837718425910333,
        'area': 1137.355369029224,
        'lat': 59.938732,
        'lon': 30.316229
    },
    'Омск': {
        'type': 'city',
        'importance': 0.706863149131586,
        'area': 2424.721167579471,
        'lat': 54.991375,
        'lon': 73.371529
    },
    'Н.Новгород': {
        'type': 'signal',
        'importance': 0.11100000000000002,
        'area': 6.734885040711671e-05,
        'lat': 57.1810745,
        'lon': 45.1382139
    },
    'Ни́жний Но́вгород': {
        'type': 'city',
        'importance': 0.972093333761689,
        'area': 570.3511705993303,
        'lat': 56.328571,
        'lon': 44.003506
    },
}


def area_city(geocode):
    lat = list(map(float, geocode.raw['boundingbox'][:2]))
    lon = list(map(float, geocode.raw['boundingbox'][2:]))
    length = distance.distance((lat[0], lon[0]), (lat[0], lon[1])).km
    breadth = distance.distance((lat[0], lon[0]), (lat[1], lon[0])).km
    return length * breadth

def add_city_name(shops_df):
    shops_df['city'] = shops_df.shop_name.apply(lambda x: x.split(' ')[0])
    shops_df.loc[shops_df.city =='РостовНаДону', 'city'] = 'Ростов На Дону'
    shops_df.loc[shops_df.city =='Н.Новгород', 'city'] = 'Ни́жний Но́вгород'


class CityData:
    def __init__(self, shops_df):
        self._shops_df = shops_df.copy()
        self.data = {}
        self._geolocator = None

        add_city_name(self._shops_df)
        self._invalid_cities = ['Интернет-магазин', 'Выездная', 'Цифровой']


    def run(self):
        if self._geolocator is None:
            self._geolocator = Nominatim(
                user_agent="sales prediction_{}".format(np.random.rand()))
        cities_ru = set(self._shops_df.city.values)
        for city in cities_ru:
            if city in self._invalid_cities or city in self.data:
                continue

            location = self._geolocator.geocode(city + ', Россия')
            assert location.raw['type'] in ['city', 'town', 'administrative']
            self.data[city] = {
                'type': location.raw['type'],
                'importance': location.raw['importance'],
                'area': area_city(location),
                'lat': location.latitude,
                'lon': location.longitude
            }

            print(city, location.raw['type'],
                  round(location.raw['importance'], 2),
                  (location.latitude, location.longitude))

