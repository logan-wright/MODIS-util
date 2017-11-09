# Purpose:
#   Find and download MODIS granules
#
# by Hong Chen (me@hongchen.cz)
#
# Tested on macOS v10.12.6 with
#   - Python v3.6.0

import os
import sys
import glob
import datetime
import numpy as np
import matplotlib.path as mpl_path
import cartopy.crs as ccrs
import ftplib

def READ_GEOMETA(date, satID='aqua', fdir='/Users/hoch4240/Chen/mygit/MODIS-util/data/geoMeta/6'):
    """
    output:
        GeoMeta data of MODIS that contains the input points, e.g. flight track.

        data['GranuleID'].decode('UTF-8') to get the file name of MODIS granule
        data['StartDateTime'].decode('UTF-8') to get the time stamp of MODIS granule

        GranuleID
        StartDateTime
        ArchiveSet
        OrbitNumber
        DayNightFlag
        EastBoundingCoord
        NorthBoundingCoord
        SouthBoundingCoord
        WestBoundingCoord
        GRingLongitude1
        GRingLongitude2
        GRingLongitude3
        GRingLongitude4
        GRingLatitude1
        GRingLatitude2
        GRingLatitude3
        GRingLatitude4
    """

    fdir = '%s/%s/%4.4d' % (fdir, satID.upper(), date.year)

    fnames = sorted(glob.glob('%s/*%s*.txt' % (fdir, datetime.datetime.strftime(date, '%Y-%m-%d'))))
    if len(fnames) == 0:
        exit('Error [READ_GEOMETA]: cannot find file under %s for %s.' % (fdir, str(date)))
    elif len(fnames) > 1:
        print('Warning [READ_GEOMETA]: find more than 1 file under %s for %s.' % (fdir, str(date)))

    fname = fnames[0]

    with open(fname, 'r') as f:
        header = f.readline()

    header = header.replace('#', '').split('\\n')
    vnames = header[-1].strip().split(',')
    dtype  =  []
    for vname in vnames:
        if vname == 'GranuleID':
            form = (vname, '|S41')
            dtype.append(form)
        elif vname == 'StartDateTime':
            form = (vname, '|S16')
            dtype.append(form)
        elif vname == 'ArchiveSet':
            form = (vname, '<i4')
            dtype.append(form)
        elif vname == 'DayNightFlag':
            form = (vname, '|S1')
            dtype.append(form)
        else:
            form = (vname, '<f8')
            dtype.append(form)

    # variable names can be found under data.dtype.names
    data = np.genfromtxt(fname, delimiter=',', skip_header=1, names=vnames, dtype=dtype)

    return data

def FIND_MODIS(date, tmhr, lon, lat, satID='aqua', percentIn_threshold=0.0, tmhr_range=None):

    """
    Input:
        date: Python datetime.datetime object
        tmhr: time in hour of, e.g. flight track
        lon : longitude of, e.g. flight track
        lat : latitude of, e.g. flight track

        satID: default "aqua", can also change to "terra"
        percentIn_threshold: default 0.0, threshold percentage of input points, e.g. flight track, within MODIS 5min granule
        tmhr_range: default None, can be set using Python list, e.g. [tmhr0, tmhr1], to define time range

    output:
        GeoMeta data of MODIS that contains the input points, e.g. flight track.

        data['GranuleID'].decode('UTF-8') to get the file name of MODIS granule
        data['StartDateTime'].decode('UTF-8') to get the time stamp of MODIS granule
    """

    lon[lon>180.0] -= 360.0
    logic  = (tmhr>=0.0)&(tmhr<48.0) & (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)

    tmhr   = tmhr[logic]
    lon    = lon[logic]
    lat    = lat[logic]

    data = READ_GEOMETA(date, satID=satID)

    # calculate tmhr (time in hour) from MODIS time stamp
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    Ndata = data.size
    tmhr_modis = np.zeros(Ndata, dtype=np.float32)
    for i in range(Ndata):
        tmhr_modis[i] = (datetime.datetime.strptime(data['StartDateTime'][i].decode('UTF-8'), '%Y-%m-%d %H:%M') - date).total_seconds() / 3600.0
    # ---------------------------------------------------------------------

    # find data within given/default time range
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if tmhr_range != None:
        try:
            indices = np.where((tmhr_modis>=tmhr_range[0]) & (tmhr_modis<tmhr_range[1]))[0]
        except IndexError:
            indices = np.where((tmhr_modis>=tmhr.min()) & (tmhr_modis<=tmhr.max()))[0]
    else:
        indices = np.arange(Ndata)
    # ---------------------------------------------------------------------

    # loop through all the "MODIS granules" constructed through four corner points
    # and find which granules contain the input data
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    proj_ori = ccrs.PlateCarree()
    indices_find   = []

    # the longitude in GeoMeta dataset is in the range of [-180, 180]
    for i, index in enumerate(indices):

        line = data[index]
        xx0  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
        yy0  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

        if (abs(xx0[0]-xx0[1])>180.0) | (abs(xx0[0]-xx0[2])>180.0) | \
           (abs(xx0[0]-xx0[3])>180.0) | (abs(xx0[1]-xx0[2])>180.0) | \
           (abs(xx0[1]-xx0[3])>180.0) | (abs(xx0[2]-xx0[3])>180.0):

            xx0[xx0<0.0] += 360.0

        # roughly determine the center of granule
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()
        # ---------------------------------------------------------------------

        # find a more precise center point of MODIS granule
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        # ---------------------------------------------------------------------

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) / float(pointsIn.size) * 100.0
        if (percentIn > percentIn_threshold):
            indices_find.append(index)
    # ---------------------------------------------------------------------

    return data[indices_find]

class FTP_INIT:

    def __init__(self, data):

        filenames = []
        for line in data:
            filenames.append(line['GranuleID'].decode('UTF-8'))

        self.filenames = filenames

        self.PRE_FTP()

    def PRE_FTP(self, dataExts=['03', '06_L2']):

        namePatterns = []
        for filename in self.filenames:
            words = filename.split('.')
            satID = words[0][:3]
            year  = words[1][1:5]
            doy   = words[1][5:]
            pattern = '.'.join(words[1:4])

            for dataExt in dataExts:
                tag = satID+dataExt
                namePattern = '/allData/6/%s/%s/%s/%s.%s' % (tag, year, doy, tag, pattern)
                namePatterns.append(namePattern)

        self.namePatterns = namePatterns

def DOWNLOAD_MODIS(ftp_init, fdirOut=os.getcwd(), verbose=True):

    """
    input:
        data: FTP_INIT object instance

    output:
        N/A
    """

    if not os.path.isdir(fdirOut):
        os.system('mkdir -p %s' % fdirOut)

    ftpSite  = 'ladsftp.nascom.nasa.gov'
    try:
        ftpMODIS = ftplib.FTP(ftpSite)
        ftpMODIS.login()
    except ftplib.all_errors:
        exit('Error [DOWNLOAD_MODIS]: cannot FTP to %s.' % ftpSite)

    for namePattern in ftp_init.namePatterns:
        words = namePattern.split('/')
        ftpFdir = '/'.join(words[:-1])
        pattern = words[-1]
        try:
            ftpMODIS.cwd(ftpFdir)
            fnames = ftpMODIS.nlst()

            count = 0
            while (pattern not in fnames[count]):
                count += 1
            fname = fnames[count]

            if os.path.exists('%s/%s' % (fdirOut, fname)):
                print('Warning [DOWNLOAD_MODIS]: %s exists under %s.' % (fname, fdirOut))
            else:
                ftpMODIS.retrbinary('RETR %s' % fname, open('%s/%s' % (fdirOut, fname), 'wb').write)
                if verbose and glob.glob('%s/%s' % (fdirOut, fname)):
                    print('Message [DOWNLOAD_MODIS]: %s has been downloaded under %s.' % (fname, fdirOut))

        except ftplib.all_errors:
            exit('Error [DOWNLOAD_MODIS]: data is not available for the requested date.')

    ftpMODIS.quit()

class READ_ICT_HSK:

    def __init__(self, date, fdir='/Users/hoch4240/Chen/work/01_ARISE/comp2/data/hsk'):

        self.date = date

        fnames = glob.glob('%s/*%s*' % (fdir, date))
        Nfiles = len(fnames)
        if Nfiles == 1:
            fname = fnames[0]
        elif Nfiles > 1:
            fname = fnames[0]
            print("Warning [READ_ICT_HSK]: found more than 1 file for %s." % date)
        else:
            exit('Error [READ_ICT_HSK]: no file found for %s' % date)

        f = open(fname, 'r')
        firstLine = f.readline()
        skip_header = int(firstLine.split(',')[0])

        vnames = []
        units  = []
        for i in range(7):
            f.readline()
        vname0, unit0 = f.readline().split(',')
        vnames.append(vname0.strip())
        units.append(unit0.strip())
        Nvar = int(f.readline())
        for i in range(2):
            f.readline()
        for i in range(Nvar):
            vname0, unit0 = f.readline().split(',')
            vnames.append(vname0.strip())
            units.append(unit0.strip())
        f.close()

        data = np.genfromtxt(fname, skip_header=skip_header, delimiter=',')

        self.data = {}

        for i, vname in enumerate(vnames):
            self.data[vname] = data[:, i]

if __name__ == '__main__':

    hsk  = READ_ICT_HSK('20140913')
    tmhr = (hsk.data['Start_UTC']/3600.0)[::10]
    lon  = hsk.data['Longitude'][::10]
    lat  = hsk.data['Latitude'][::10]

    date = datetime.datetime(2014, 9, 13)
    data = FIND_MODIS(date, tmhr, lon, lat, satID='terra')
    ftp_init = FTP_INIT(data)
    DOWNLOAD_MODIS(ftp_init)
