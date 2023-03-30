"""Region definitions.

Functions
---------

"""


def get_region_dict(region=None):
    regions_dict = {

        "globe": {
            "lat_range": [-90, 90],
            "lon_range": [0, 360],
        },

        "northern hemisphere": {
            "lat_range": [0, 90],
            "lon_range": [0, 360],
        },

        "southern hemisphere": {
            "lat_range": [-90, 0],
            "lon_range": [0, 360],
        },

        "north atlantic": {
            "lat_range": [40, 60],
            "lon_range": [360 - 70, 360 - 10],
        },

        "eastern europe": {
            "lat_range": [40, 60],
            "lon_range": [0, 30],
        },

        "western us": {
            "lat_range": [30, 49],
            "lon_range": [360-125, 360-110],
        },

        "india": {
            "lat_range": [10, 30],
            "lon_range": [70, 85],
        },

        "north pdo": {
            "lat_range": [48., 60.],
            "lon_range": [360-165, 360-124],
        },

        "nino34" :{
            "lat_range":[-5., 5.],
            "lon_range":[360-170, 360-120],
        },

        "indopac" :{
            "lat_range":[-30., 30.],
            "lon_range":[30, 360-80],
        },

        "sams" :{
            "lat_range":[-20., -5.],
            "lon_range":[300., 320.],

        },

        "southern europe" :{
            "lat_range":[29., 49.],
            "lon_range":[16., 36.],
        },

        "n_atlantic" :{
            "lat_range":[0., 65.],
            "lon_range":[360.-70., 360.],
        },

        "trop_pac_precip" :{
            "lat_range":[-5., 5.],
            "lon_range":[170., 200.],
        },        


    }

    if region is None:
        return regions_dict
    else:
        return regions_dict[region]
