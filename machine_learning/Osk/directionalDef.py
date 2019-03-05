import numpy as np
import astropy as apy
from astropy import units as u
import astropy.coordinates, astropy.time
import datetime
import timeit
import matplotlib.pyplot as plt
import pyne2001

pyne2001.test()

start = timeit.default_timer()

def lovellOffsets(AZ, EL):

    ia = 0.641389
    ie = 1.129125
    npae = 0.020131
    ca = 0.
    an = 0.003919
    aw = 0.
    tf = 0.213564
    haca2 = 0.0
    hasa2 = 0.0
    cote = 0.017222

    a = ia
    b = haca2 * (np.cos(AZ)**2)
    c = hasa2 * (-1 * (np.sin(AZ)**2))
    d = an * np.sin(AZ) * np.sin(EL)
    e = -1 * aw * (-1 * np.cos(AZ)) * (np.sin(EL))
    f = npae * np.sin(EL)
    g = ca/(np.cos(EL))

    deltaAZ = -(a + b + c + (d + e + f + g))

    h = ie
    i = an * (-1 * np.cos(AZ))
    j = aw * np.sin(AZ)
    k = -1 * (tf * np.cos(EL))

    deltaEL = h + i + j + k

    return deltaAZ, deltaEL


def getEquitorialCoordinates(encoderAzimuth, encoderElevation, azimuthOffset, elevationOffset, date):

    telescopeLocations = {'lovell':[3822252.6430, -153995.6830, 5086051.4430]}
    lovellLocation = apy.coordinates.EarthLocation.from_geocentric(telescopeLocations['lovell'][0] * u.m, telescopeLocations['lovell'][1] * u.m, telescopeLocations['lovell'][2] * u.m)
    if date == 0:
        time = apy.time.Time(datetime.datetime.utcnow())
    else:
        time = apy.time.Time(date)

    if float(azimuthOffset) > 180:
        azimuthOffset = float(azimuthOffset) - 360
    if float(elevationOffset) > 180:
        elevationOffset = float(elevationOffset) - 360

    # Apply offset
    azimuth = float(encoderAzimuth) - float(azimuthOffset)
    altitude = float(encoderElevation) - float(elevationOffset)

    if altitude > 90:
        altitude = 180 - altitude
        azimuth = azimuth + 180

    # Feed into astropy
    azimuthElevation = apy.coordinates.AltAz(az = azimuth * u.deg, alt = altitude * u.deg, location = lovellLocation, obstime = time)
    #convert to ra and dec
    raDec = apy.coordinates.SkyCoord(azimuthElevation.transform_to(apy.coordinates.ICRS))
    raDec
    equitorialCoordinates = raDec.to_string('hmsdms')
    return equitorialCoordinates

stop = timeit.default_timer()

print('Time: ', stop - start, "\n")

print (lovellOffsets(170.001, 94.694))
print (getEquitorialCoordinates(170.001, 94.694, lovellOffsets(170.001, 94.694)[0], lovellOffsets(170.001, 94.694)[1], '2019-02-13 10:30:00'))

#AZ_arr=[]
#EL_arr=[]
#for i in range(80, 100):
#    AZoffset, ELoffset = lovellOffsets(170, i)
#    AZ_arr.append(AZoffset)
#    EL_arr.append(ELoffset)

#plt.plot(EL_arr)