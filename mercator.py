
import math
import numpy as np

earthRadius = 6371008.8
earthCircumference = 2 * math.pi * earthRadius

def circumferenceAtLatitude(latitude):
    return earthCircumference * math.cos(latitude * math.pi / 180)

def mercatorXfromLng(lng):
    return (180 + lng) / 360

def mercatorYfromLat(lat):
    return (180 - (180 / math.pi * math.log(math.tan(math.pi / 4 + lat * math.pi / 360)))) / 360

def mercatorZfromAltitude(altitude, lat):
    return altitude / circumferenceAtLatitude(lat)

def lat_lon_to_mercator(ref_lat, ref_lon):
    """
    Convert reference center point's longitude and latitude to Mercator coordinates
    """
    x = mercatorXfromLng(ref_lon)
    y = mercatorYfromLat(ref_lat)
    return x, y

def xyz_to_mercator(x, y, z, ref_mercator_x, ref_mercator_y, mercator_constant):
    """
    Convert point's xyz coordinates to Mercator coordinates
    """
    # Calculate new Mercator coordinates
    new_x = ref_mercator_x + x * mercator_constant
    new_y = ref_mercator_y - y * mercator_constant

    return new_x, new_y

def mercator_zoom_n(zoom):
    return 2 ** zoom

def mercator_to_tile_id(mercator_x, mercator_y, zoom_n):
    """
    Convert Mercator coordinates to Google tile x, y coordinates
    """
    x = int(mercator_x * zoom_n)
    y = int(mercator_y * zoom_n)
    return x, y


def geodetic_to_ecef_transformation(longitude, latitude, height=0, r=6378137.0, f=1/298.257223563):
    # Convert degrees to radians
    phi = math.radians(latitude)
    lam = math.radians(longitude)
    
    # Calculate radius of curvature of the ellipsoid
    e2 = 2 * f - f ** 2
    N = r / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    
    # Calculate ECEF coordinates
    x = (N + height) * math.cos(phi) * math.cos(lam)
    y = (N + height) * math.cos(phi) * math.sin(lam)
    z = (N * (1 - e2) + height) * math.sin(phi)
    
    # Construct transformation matrix

    transformation_matrix = np.array([        
        -math.sin(lam), math.cos(lam), 0, 0,
        -math.sin(phi)*math.cos(lam), -math.sin(phi)*math.sin(lam), math.cos(phi), 0,
        math.cos(phi)*math.cos(lam), math.cos(phi)*math.sin(lam), math.sin(phi), 0,
        x, y, z, 1
    ])

    
    return transformation_matrix
