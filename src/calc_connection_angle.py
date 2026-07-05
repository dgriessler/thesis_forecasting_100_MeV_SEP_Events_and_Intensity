import csv
import math
from astropy.time import Time

def readCSVFile(csvFile):
    rows = []
    with open(csvFile, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    
    return rows

def parseDateTime(val):
    date_time = val.split(' ')
    month_day_year = date_time[0].split('/')
    hour_minute_second = date_time[1].split(':')
    if len(hour_minute_second) < 3:
        second = 0
    else:
        second = hour_minute_second[2]
    date_data = {
        "month": int(month_day_year[0]),
        "day" : int(month_day_year[1]),
        "year" : int(month_day_year[2]),
        "hour" : int(hour_minute_second[0]),
        "minute" : int(hour_minute_second[1]),
        "second" : int(second)
    }
    return Time(date_data, scale="utc")

def calculate_connection_angle(latitude, longitude, solar_wind_speed):
    angular_speed_of_sun = 360 / (27.27 * 86400)
    au = 1.5 * math.pow(10, 8)
    # connection angle = arccos(sin(theta_1) * sin(theta_2) + cos(theta_1) * cos(theta_2) * cos(phi_1 - phi_2))
    # theta_1 = latitude (in degrees) up and down (parallel to prime meridian)
    # phi_1 = longitude (in degrees) right and left (parallel to equator)
    # theta_2 = 0
    # phi_2 = angular_speed_of_sun * 1 AU / solar_wind_speed (in degrees)
    # angular_speed_of_sun = 360 / 27.27 * 86400) degrees/second
    # 1 AU = 1.5 * 10^8 km
    theta_1 = float(latitude)
    phi_1 = float(longitude)
    theta_2 = 0
    phi_2 = angular_speed_of_sun * au / float(solar_wind_speed)
    # math.acos wants RADIANS not DEGREES
    # math.sin wants RADIANS not DEGREES
    # math.cos wants RADIANS not DEGREES
    theta_1_rad = math.radians(theta_1)
    phi_1_rad = math.radians(phi_1)
    theta_2_rad = math.radians(theta_2)
    phi_2_rad = math.radians(phi_2)
    # math.acos only returns positive values. The connection angle sometimes should be negative
    if phi_1 >= phi_2 - 180 and phi_1 <= phi_2:
        sign = -1
    else:
        sign = 1

    # math.acos returns RADIANS but we want DEGREES
    connection_angle_rad = sign * math.acos(math.sin(theta_1_rad) * math.sin(theta_2_rad) + math.cos(theta_1_rad) * math.cos(theta_2_rad) * math.cos(phi_1_rad - phi_2_rad))

    return math.degrees(connection_angle_rad)

def calculate_sanity_check(latitude, longitude, solar_wind_speed):
    sanity_check = float(longitude) - 57
    if sanity_check < -180:
        sanity_check = sanity_check + 360

    return sanity_check

if __name__ == "__main__":
    data = readCSVFile("../res/adapted_rRT_data_learn_richardson.csv")

    max_problem = None
    max_problem_val = None
    for elem in data:
        longitude = float(elem["longitude"])
        latitude = float(elem["latitude"])
        solar_wind_speed = float(elem["solar_wind_speed"])
        connection_angle = calculate_connection_angle(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)
        sanity_check = calculate_sanity_check(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)
        diff = abs(connection_angle - sanity_check)
        if diff > 30:
            print("DONKI_DATE: {}. LONG: {}. LAT: {}. SOLAR_WIND_SPEED: {}. CONN_ANG_DEG: {}. Sanity Check: {}. ABS(Conn_ang_deg - sanity_check): {}".format(elem["donki_date"], longitude, latitude, solar_wind_speed, connection_angle, sanity_check, abs(connection_angle - sanity_check)))
            if max_problem_val is None or diff > max_problem_val:
                max_problem = elem
                max_problem_val = diff

    if max_problem is not None:
        print("\n\nMAX PROBLEM\n")
        longitude = float(max_problem["longitude"])
        latitude = float(max_problem["latitude"])
        solar_wind_speed = float(max_problem["solar_wind_speed"])
        connection_angle = calculate_connection_angle(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)
        sanity_check = calculate_sanity_check(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)

        angular_speed_of_sun = 360 / (27.27 * 86400)
        au = 1.5 * math.pow(10, 8)
        phi_1 = float(longitude)
        phi_2 = angular_speed_of_sun * au / float(solar_wind_speed)

        print("PHI_1: {}. PHI_2: {}.".format(phi_1, phi_2))
        print("DONKI_DATE: {}. LONG: {}. LAT: {}. SOLAR_WIND_SPEED: {}. CONN_ANG_DEG: {}. Sanity Check: {}. ABS(Conn_ang_deg - sanity_check): {}".format(max_problem["donki_date"], longitude, latitude, solar_wind_speed, connection_angle, sanity_check, abs(connection_angle - sanity_check)))

        print("\n\nSUB PROBLEM\n")
        longitude = 45.0
        latitude = 42.0
        solar_wind_speed = 519.0
        connection_angle = calculate_connection_angle(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)
        sanity_check = calculate_sanity_check(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)

        angular_speed_of_sun = 360 / (27.27 * 86400)
        au = 1.5 * math.pow(10, 8)
        phi_1 = float(longitude)
        phi_2 = angular_speed_of_sun * au / float(solar_wind_speed)
        print("PHI_1: {}. PHI_2: {}.".format(phi_1, phi_2))

        print("DONKI_DATE: 12/16/2010 9:12. LONG: {}. LAT: {}. SOLAR_WIND_SPEED: {}. CONN_ANG_DEG: {}. Sanity Check: {}. ABS(Conn_ang_deg - sanity_check): {}".format(longitude, latitude, solar_wind_speed, connection_angle, sanity_check, abs(connection_angle - sanity_check)))



        print("\n\ORIG PROBLEM\n")
        longitude = -96.0
        latitude = 86.0
        solar_wind_speed = 271.8
        connection_angle = calculate_connection_angle(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)
        sanity_check = calculate_sanity_check(latitude=latitude, longitude=longitude, solar_wind_speed=solar_wind_speed)

        angular_speed_of_sun = 360 / (27.27 * 86400)
        au = 1.5 * math.pow(10, 8)
        phi_1 = float(longitude)
        phi_2 = angular_speed_of_sun * au / float(solar_wind_speed)
        print("PHI_1: {}. PHI_2: {}.".format(phi_1, phi_2))

        print("LONG: {}. LAT: {}. SOLAR_WIND_SPEED: {}. CONN_ANG_DEG: {}. Sanity Check: {}. ABS(Conn_ang_deg - sanity_check): {}".format(longitude, latitude, solar_wind_speed, connection_angle, sanity_check, abs(connection_angle - sanity_check)))

