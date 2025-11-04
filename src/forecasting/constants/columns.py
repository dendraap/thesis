# Initialize encode columns name
col_encode = {
    "timestamp"                 : "t",
    "pm2_5 (μg/m³)"             : "y1",
    "pm10 (μg/m³)"              : "y2",
    "carbon_monoxide (μg/m³)"   : "y3",
    "nitrogen_dioxide (μg/m³)"  : "y4",
    "sulphur_dioxide (μg/m³)"   : "y5",
    "ozone (μg/m³)"             : "y6",
    "temperature_2m (°C)"       : "x1",
    "relative_humidity_2m (%)"  : "x2",
    "dew_point_2m (°C)"         : "x3",
    "rain (mm)"                 : "x4",
    "surface_pressure (hPa)"    : "x5",
    "wind_speed_10m (km/h)"     : "x6",
    "wind_direction_10m (°)"    : "x7",
    "direct_radiation (W/m²)"   : "x8",
    "pembagian_waktu/hari"      : "x9",
    "hari_per_minggu"           : "x10",
    "hari_libur_nasional"       : "x11",
    "musim"                     : "x12"
}

# Initialize decode columns name
col_decode = {
    "t"  : "timestamp",
    "y1" : "pm2_5 (μg/m³)",
    "y2" : "pm10 (μg/m³)",
    "y3" : "carbon_monoxide (μg/m³)",
    "y4" : "nitrogen_dioxide (μg/m³)",
    "y5" : "sulphur_dioxide (μg/m³)",
    "y6" : "ozone (μg/m³)",
    "x1" : "temperature_2m (°C)",
    "x2" : "relative_humidity_2m (%)",
    "x3" : "dew_point_2m (°C)",
    "x4" : "rain (mm)",
    "x5" : "surface_pressure (hPa)",
    "x6" : "wind_speed_10m (km/h)",
    "x7" : "wind_direction_10m (°)",
    "x8" : "direct_radiation (W/m²)",
    "x9" : "pembagian_waktu/hari",
    "x10": "hari_per_minggu",
    "x11": "hari_libur_nasional",
    "x12": "musim"
}

# Initialize encoder categorical columns to ordinal ranking
waktu_hari_ordinal = {'Pagi': '1', 'Siang': '2', 'Malam': '3'}
hari_ordinal = {'Monday': '1', 'Tuesday': '2', 'Wednesday': '3', 'Thursday': '4', 'Friday': '5', 'Saturday': '6', 'Sunday': '7'}
musim_ordinal = {'Musim kemarau': '1', 'Musim hujan': '2'}