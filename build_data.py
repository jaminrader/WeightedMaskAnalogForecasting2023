"""Data processing.

Functions
---------

"""

import numpy as np
import regions
import xarray as xr
import metrics
import base_directories
import pickle
import gzip
import os
import warnings

__author__ = "Jamin K. Rader, Elizabeth A. Barnes, and Randal J. Barnes"
__version__ = "30 March 2023"

dir_settings = base_directories.get_directories()


# def batch_generator_static(settings, soi_input, soi_output, analog_input, analog_output):
#     data_input = np.concatenate((soi_input, analog_input), axis=0)
#     data_output = np.concatenate((soi_output, analog_output), axis=0)
#
#     yield [data_input, data_input], [data_output]


def batch_generator(settings, soi_input, soi_output, analog_input, analog_output, batch_size, rng_seed=33):

    rng = np.random.default_rng(rng_seed)

    while True:
        i_soi = rng.choice(np.arange(0, soi_input.shape[0]), batch_size, replace=True)
        i_analog = rng.choice(np.arange(0, analog_input.shape[0]), batch_size, replace=True)
        targets = metrics.get_targets(settings, soi_output[i_soi], analog_output[i_analog])
        yield [soi_input[i_soi, :, :, :], analog_input[i_analog, :, :, :]], [targets]


def maskout_land_ocean(da, maskout="land"):
    # if no land mask or ocean masks exists, run make_land_ocean_mask()
    if maskout == "land":
        with gzip.open(dir_settings["data_directory"] + "MPI-ESM_ocean_mask.pickle", "rb") as fp:
            mask = pickle.load(fp)
    elif maskout == "ocean":
        with gzip.open(dir_settings["data_directory"] + "MPI-ESM_land_mask.pickle", "rb") as fp:
            mask = pickle.load(fp)
    # if maskout == "land":
    #     mask = xr.load_dataarray(dir_settings["data_directory"] + "MPI-ESM_ocean_mask.nc").to_numpy()
    # elif maskout == "ocean":
    #     mask = xr.load_dataarray(dir_settings["data_directory"] + "MPI-ESM_land_mask.nc").to_numpy()
    else:
        raise NotImplementedError("no such mask type.")
    return da*mask


def build_data(settings, data_directory):

    # Use experiment name or base experiment name for data?
    if "base_exp_name" in settings.keys():
        data_exp_name = settings["base_exp_name"]
    else:
        data_exp_name = settings["exp_name"]

    if settings["presaved_data_filename"] is None:
        data_savename = dir_settings["data_directory"] + 'presaved_data_' + data_exp_name + '.pickle'
    else:
        data_savename = dir_settings["data_directory"]+settings["presaved_data_filename"]

    if os.path.exists(data_savename) is False:
        print('building the data from netcdf files')

        # initialize empty dictionaries to old the standardization info from the training data
        input_standard_dict = {
            "ens_mean": None,
            "data_mean": None,
            "data_std": None,
        }
        output_standard_dict = {
            "ens_mean": None,
            "data_mean": None,
            "data_std": None,
        }
        print('getting analog pool...')
        analog_input, analog_output, input_standard_dict, output_standard_dict = process_input_output(
            data_directory, settings, members=settings["analog_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )

        print('getting soi training data...')
        soi_train_input, soi_train_output, __, __ = process_input_output(
            data_directory, settings, members=settings["soi_train_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )

        print('getting validation data...')
        soi_val_input, soi_val_output, __, __ = process_input_output(
            data_directory, settings, members=settings["soi_val_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )

        print('getting testing data...')
        soi_test_input, soi_test_output, __, __ = process_input_output(
            data_directory, settings, members=settings["soi_test_members"], input_standard_dict=input_standard_dict,
            output_standard_dict=output_standard_dict,
        )

        # stack the data to dimensions of samples
        analog_input, analog_output = stack_to_samples(analog_input, analog_output)
        soi_train_input, soi_train_output = stack_to_samples(soi_train_input, soi_train_output)
        soi_val_input, soi_val_output = stack_to_samples(soi_val_input, soi_val_output)
        soi_test_input, soi_test_output = stack_to_samples(soi_test_input, soi_test_output)

        lat = analog_input["lat"]
        lon = analog_input["lon"]

        area_weights = np.cos(np.deg2rad(lat)).to_numpy()

        # save the data
        print(f"saving the pre-saved training/validation/testing data.")
        print(f"   {data_savename}")
        with gzip.open(data_savename, "wb") as fp:
            pickle.dump(analog_input.to_numpy().astype(np.float32), fp)
            pickle.dump(analog_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_train_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_train_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_val_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_val_output.to_numpy().astype(np.float32), fp)

            pickle.dump(soi_test_input.to_numpy().astype(np.float32), fp)
            pickle.dump(soi_test_output.to_numpy().astype(np.float32), fp)

            pickle.dump(input_standard_dict, fp)
            pickle.dump(output_standard_dict, fp)

            pickle.dump(lat.to_numpy().astype(np.float32), fp)
            pickle.dump(lon.to_numpy().astype(np.float32), fp)

            pickle.dump(area_weights.astype(np.float32), fp)

    # load the presaved data data
    print(f"loading the pre-saved training/validation/testing data.")
    print(f"   {data_savename}")
    with gzip.open(data_savename, "rb") as fp:
        analog_input = pickle.load(fp)
        analog_output = pickle.load(fp)

        soi_train_input = pickle.load(fp)
        soi_train_output = pickle.load(fp)

        soi_val_input = pickle.load(fp)
        soi_val_output = pickle.load(fp)

        soi_test_input = pickle.load(fp)
        soi_test_output = pickle.load(fp)

        input_standard_dict = pickle.load(fp)
        output_standard_dict = pickle.load(fp)

        lat = pickle.load(fp)
        lon = pickle.load(fp)

        area_weights = pickle.load(fp)

    # summarize the data
    analog_text = ("   analog data\n"
                   f" # analog samples = {analog_output.shape[0]}\n"
                   )
    train_text = ("   training data\n"
                  f" # soi samples = {soi_train_output.shape[0]}\n"
                  )
    val_text = ("   validation data\n"
                f"   # soi samples = {soi_val_output.shape[0]}\n"
                )
    test_text = ("   testing data\n"
                 f"  # soi samples = {soi_test_output.shape[0]}\n"
                 )
    print(analog_text + train_text + val_text + test_text)

    return (analog_input, analog_output, soi_train_input, soi_train_output, soi_val_input, soi_val_output,
            soi_test_input, soi_test_output, input_standard_dict, output_standard_dict, lat, lon, )

# Building observational data
def build_obs_data(settings, data_directory, obs_info):
    # initialize empty dictionaries to old the standardization info from the training data
    input_standard_dict = obs_info['input_standard_dict']
    output_standard_dict = obs_info['output_standard_dict']
    print('getting observations...')
    obs_input, obs_output, input_standard_dict, output_standard_dict = process_input_output(
        data_directory, settings, input_standard_dict, output_standard_dict, obs_info=obs_info
    )

    obs_input, obs_output = stack_to_samples(obs_input, obs_output)

    lat = obs_input["lat"].to_numpy().astype(np.float32) 
    lon = obs_input["lon"].to_numpy().astype(np.float32) 

    area_weights = np.cos(np.deg2rad(lat))

    return obs_input.to_numpy().astype(np.float32), \
        obs_output.to_numpy().astype(np.float32), \
        input_standard_dict, output_standard_dict, lat, lon

def process_input_output(data_directory, settings, input_standard_dict=None, 
                         output_standard_dict=None, obs_info=None, members=None):

    if obs_info == None:
        data_target = get_annual_seasonal_netcdf(settings["target_var"], data_directory, settings, members=members)
        data_feature = get_annual_seasonal_netcdf(settings["feature_var"], data_directory, settings, members=members)
    else:
        data_target = get_annual_seasonal_netcdf(obs_info["target_var"], data_directory, 
                                                 settings, obs_fn=obs_info["target_filename"])
        data_feature = get_annual_seasonal_netcdf(obs_info["feature_var"], data_directory, 
                                                  settings, obs_fn=obs_info["feature_filename"])

    # PROCESS TARGET DATA
    # turn output region data into a single scalar by averaging the results over the region
    if settings["maskout_landocean_output"] is not None:
        data_target = maskout_land_ocean(data_target, settings["maskout_landocean_output"])
    data_output, __, __ = extract_region(data_target, regions.get_region_dict(settings["target_region_name"]))
    data_output = compute_global_mean(data_output)

    # PROCESS FEATURE DATA
    if settings["maskout_landocean_input"] is not None:
        data_feature = maskout_land_ocean(data_feature, settings["maskout_landocean_input"])
    (data_input, data_output, input_standard_dict, output_standard_dict) = process_data(
        data_feature, data_output, settings, input_standard_dict, output_standard_dict)

    return data_input, data_output, input_standard_dict, output_standard_dict


def stack_to_samples(data_input, data_output):
    return (data_input.stack(sample=("member", "year")).transpose("sample", "lat", "lon", "channel", ), 
            data_output.stack(sample=("member", "year")).transpose( "sample", ))


def repeat_soi_members(soi_input, soi_output, n_members_analog):
    # check that the soi have the same number of members as the analog by copying
    if soi_input["member"].shape[0] != n_members_analog:
        n_members_soi = soi_input["member"].shape[0]
        print(f"  {n_members_soi} members in soi != {n_members_analog} members in analog, repeating soi members...")
        print(f"  before soi_data.shape = {soi_input.shape}")

        assert n_members_analog % n_members_soi == 0, "number of members in the analog set is not divisible by the " \
                                                      "number of members in the soi set"

        n_repeats = n_members_analog / n_members_soi
        soi_input_repeat = soi_input
        soi_output_repeat = soi_output
        for icycle in np.arange(0, n_repeats - 1):
            soi_input_repeat = xr.concat((soi_input_repeat, soi_input), dim="member")
            soi_output_repeat = xr.concat((soi_output_repeat, soi_output), dim="member")

        print(f"  after soi_data.shape = {soi_input_repeat.shape}")
        return soi_input_repeat, soi_output_repeat
    else:
        return soi_input, soi_output


def process_data(data_feature, data_target, settings, input_standard_dict, output_standard_dict):
    # smooth the data and add tendency channel
    data_input = smooth_data(data_feature, settings["smooth_len_input"])
    data_input = add_extra_channel(data_input, settings)
    data_output = smooth_data(data_target, settings["smooth_len_output"])

    # only grab valid years in input and output data sets
    # then shift the data to have a lead/lag
    data_input, data_output = xr.align(data_input, data_output, join="inner", copy=True)
    data_input, data_output = create_input_output_shift(data_input, data_output, settings["lead_time"])
    
    # remove the forced components (ensemble-mean)
    data_input, input_standard_dict = remove_ensemble_mean(data_input, input_standard_dict)
    data_output, output_standard_dict = remove_ensemble_mean(data_output, output_standard_dict)

    # standardize the data according to the training data
    data_input, input_standard_dict = standardize_data(data_input, input_standard_dict, settings["standardize_bool"])
    data_output, output_standard_dict = standardize_data(data_output, output_standard_dict,
                                                         settings["standardize_bool"])

    return data_input, data_output, input_standard_dict, output_standard_dict


def add_extra_channel(data_in, settings):
    if settings["extra_channel"] == None or settings["extra_channel"] == 0:
        return data_in.expand_dims(dim={"channel": 1}, axis=-1).copy()
    else:
        d_present, d_past = xr.align(data_in[:, settings["extra_channel"]:, :, :],
                                     data_in[:, :-settings["extra_channel"], :, :], join="override", copy=True)
        data_in = d_present.expand_dims(dim={"channel": 2}, axis=-1).copy()
        data_in[:, :, :, :, 1] = data_in[:, :, :, :, 0] - d_past

        return data_in


def create_input_output_shift(data_input, data_output, lead_time):
    if lead_time == 0:
        return data_input.copy(), data_output.copy()
    else:
        data_in = data_input[:, :-lead_time, :, :].copy()
        data_out = data_output[:, lead_time:,].copy()
        return data_in, data_out


def smooth_data(data, smooth_time):
    if smooth_time == 0:
        return data

    if smooth_time > 0:
        return data.rolling(year=smooth_time, center=False).mean().dropna("year")
    if smooth_time < 0:
        data_roll = data.rolling(year=np.abs(smooth_time), center=False).mean().dropna("year").copy()
        return data_roll.assign_coords(year=data_roll["year"].values - (np.abs(smooth_time) - 1))


def compute_global_mean(data, lat=None):
    if isinstance(data, xr.DataArray):
        weights = np.cos(np.deg2rad(data.lat))
        weights.name = "weights"
        temp_weighted = data.weighted(weights)
        global_mean = temp_weighted.mean(("lon", "lat"), skipna=True)
    else:
        assert len(np.shape(data)) == 4, "excepted np.array of len(shape)==4)"
        weights = np.cos(np.deg2rad(lat))
        sum_weights = np.nansum(np.ones((data.shape[1], data.shape[2]))*weights[:, np.newaxis])
        global_mean = np.nansum(data*weights[np.newaxis, :, np.newaxis, np.newaxis], axis=(1, 2))/sum_weights

    return global_mean


def standardize_data(data, standard_dict, standardize_bool):

    if standard_dict["data_mean"] is None:
        if standardize_bool:
            standard_dict["data_mean"] = data.mean(axis=(0, 1)).to_numpy().astype(np.float32)
        else:
            standard_dict["data_mean"] = 0.

    if standard_dict["data_std"] is None:
        if standardize_bool:
            standard_dict["data_std"] = data.std(axis=(0, 1)).to_numpy().astype(np.float32)
        else:
            standard_dict["data_std"] = 1.0

    standardized_data = ((data - standard_dict["data_mean"]) / standard_dict["data_std"]).fillna(0.)

    return standardized_data, standard_dict


def remove_ensemble_mean(data, standard_dict=None):
    if standard_dict["ens_mean"] is None:
        standard_dict["ens_mean"] = data.mean("member")
    return data - standard_dict["ens_mean"], standard_dict


def extract_region(data, region=None, lat=None, lon=None):
    if region is None:
        min_lon, max_lon = [0, 360]
        min_lat, max_lat = [-90, 90]
    else:
        min_lon, max_lon = region["lon_range"]
        min_lat, max_lat = region["lat_range"]

    if isinstance(data, xr.DataArray):
        mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
        mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
        return data.where(mask_lon & mask_lat, drop=True), None, None
    else:
        assert len(data.shape) == 4, "expected np.array of len(shape)==4"
        ilon = np.where((lon >= min_lon) & (lon <= max_lon))[0]
        ilat = np.where((lat >= min_lat) & (lat <= max_lat))[0]
        data_masked = data[:, ilat, :, :]
        data_masked = data_masked[:, :, ilon, :]
        return data_masked, lat[ilat], lon[ilon]


def get_annual_seasonal_netcdf(var, data_directory, settings, members = None, obs_fn = None):

    if obs_fn is None:
        da_all = None

        for ens in members:

            member_text = f'{ens + 1:03}'
            print('   ensemble member = ' + member_text)

            if var == "mrsos":
                realm = 'Lmon'
            else:
                realm = 'Amon'

            if settings["scenario"] == "historical" or settings["scenario"] == "both":
                filename_hist = data_directory + var + '_' + realm + '_MPI-ESM_historical_r' + member_text + 'i1850p3_185001-200512.nc'
                dah = xr.open_dataset(filename_hist)[var].squeeze()
                dah = dah.resample(
                    time='1M'
                ).mean()  # issues with different ms timestamps
                da = dah
            if settings["scenario"] == "rcp45" or settings["scenario"] == "both":
                filename_ssp = data_directory + var + '_' + realm + '_MPI-ESM_rcp45_r' + member_text + 'i2005p3_200601-209912.nc'
                das = xr.open_dataset(filename_ssp)[var].squeeze()
                das = das.resample(
                    time='1M'
                ).mean()  # issues with different ms timestamps
                da = das
            if settings["scenario"] == "both":
                da = xr.concat([dah, das], "time")

            if da_all is None:
                da_all = da.expand_dims(dim={"member": 1}, axis=0)
            else:
                da_all = xr.concat([da_all, da], dim="member")

    else: # doing this for observations
        filename_obs = data_directory + obs_fn
        da_all = xr.open_dataset(filename_obs)[var].squeeze()
        da_all = da_all.resample(
            time='1M'
        ).mean()  # issues with different ms timestamps
        da_all = da_all.expand_dims(dim={"member": 1}, axis=0)

    da_all, __, __ = extract_region(
        data=da_all, region=regions.get_region_dict(
            settings["feature_region_name"]
        )
    )

    ### If a certain season...
    if 'season' in settings.keys():
        num_mos, cen_mo = settings["season"]
        da_all = da_all.rolling(min_periods=num_mos, center=True, time=num_mos).mean().dropna("time", how='all')
        da_all = da_all.sel(time=da_all.time.dt.month == cen_mo)
        
    return da_all.groupby('time.year').mean('time')

def make_land_mask():
    da = xr.load_dataset(dir_settings["data_directory"] +
                        "mrsos_Lmon_MPI-ESM_historical_r001i1850p3_185001-200512.nc",
                        )["mrsos"]
    x_data = da.sum(dim="time", skipna=False)
    x_ocean = xr.where(x_data >= 0.0, 0.0, 1.0)
    x_ocean.to_netcdf(dir_settings["data_directory"] + "MPI-ESM_ocean_mask.nc")
    x_ocean.plot()

    x_land = xr.where(x_data >= 0.0, 1.0, 0.0)
    x_land.to_netcdf(dir_settings["data_directory"] + "MPI-ESM_land_mask.nc")
    x_land.plot()

    da = xr.load_dataarray(dir_settings["data_directory"] + "MPI-ESM_land_mask.nc")
    data_savename = dir_settings["data_directory"] + "MPI-ESM_land_mask.pickle"
    with gzip.open(data_savename, "wb") as fp:
        pickle.dump(da.to_numpy().astype(np.float32), fp)

    da = xr.load_dataarray(dir_settings["data_directory"] + "MPI-ESM_ocean_mask.nc")
    data_savename = dir_settings["data_directory"] + "MPI-ESM_ocean_mask.pickle"
    with gzip.open(data_savename, "wb") as fp:
        pickle.dump(da.to_numpy().astype(np.float32), fp)

    return x_ocean
