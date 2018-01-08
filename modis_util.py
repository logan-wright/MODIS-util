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
from pyhdf.SD import SD, SDC
from scipy import interpolate

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

def EARTH_VIEW(data, tmhr, lon, lat):

    """
    Purpose:
        Plot input geo info and MODIS granule on map (globe).

    input:
        data: geoMeta data

        tmhr: -
        lon : --> input geo info, e.g., flight track
        lat : -
    """

    lon[lon>180.0] -= 360.0
    logic  = (tmhr>=0.0)&(tmhr<48.0) & (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)

    tmhr   = tmhr[logic]
    lon    = lon[logic]
    lat    = lat[logic]

    rcParams['font.size'] = 8.0

    proj_ori = ccrs.PlateCarree()
    for i, line in enumerate(data):

        xx0  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
        yy0  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

        if (abs(xx0[0]-xx0[1])>180.0) | (abs(xx0[0]-xx0[2])>180.0) | \
           (abs(xx0[0]-xx0[3])>180.0) | (abs(xx0[1]-xx0[2])>180.0) | \
           (abs(xx0[1]-xx0[3])>180.0) | (abs(xx0[2]-xx0[3])>180.0):

            xx0[xx0<0.0] += 360.0

        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()

        # second attempt to find the center point of MODIS granule
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        # ---------------------------------------------------------------------

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        ax = plt.axes(projection=proj_new)
        ax.set_global()
        ax.stock_img()
        ax.coastlines(color='gray', lw=0.2)
        title = RENAME_MODIS(data[i]['GranuleID'].decode('UTF-8'))
        ax.set_title(title, fontsize=8)

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) / float(pointsIn.size) * 100.0
        if (percentIn > 0):
            patch = patches.PathPatch(modis_granule, facecolor='g', edgecolor='g', alpha=0.4, lw=0.2)
        else:
            patch = patches.PathPatch(modis_granule, facecolor='k', edgecolor='k', alpha=0.2, lw=0.2)

        cs = ax.scatter(lon, lat, transform=ccrs.Geodetic(), s=0.01, c=tmhr, cmap='jet')
        ax.scatter(xx.mean(), yy.mean(), marker='*', transform=ccrs.Geodetic(), s=6, c='r')
        ax.scatter(center_lon, center_lat, marker='*', transform=ccrs.Geodetic(), s=6, c='b')
        ax.add_patch(patch)
        plt.colorbar(cs, shrink=0.6)
        plt.savefig('%s.png' % '.'.join(title.split('.')[:-1]))
        plt.close()

    # ---------------------------------------------------------------------

def RENAME_MODIS(filename):
    """
    Purpose:
        Change the "Day-Of-Year" in the MODIS file name to "YearMonthDay"

    input:
        MODIS file name, e.g. MYD03.A2014290.0035.006.2014290162522.hdf
    output:
        new MODIS file name, e.g., MYD03.A20141017.0035.006.2014290162522.hdf
    """

    try:
        fwords = filename.split('.')
        date = datetime.datetime.strptime(fwords[1], 'A%Y%j')
        fwords[1] = date.strftime('A%Y%m%d')
        return '.'.join(fwords)

    except ValueError:
        print('Warning [RENAME_MODIS]: cannot convert, return input filename as new filename.')
        return filename

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
            fname  = fnames[count]
            fname_new = RENAME_MODIS(fname)

            if os.path.exists('%s/%s' % (fdirOut, fname)):
                print('Warning [DOWNLOAD_MODIS]: %s exists under %s.' % (fname, fdirOut))
            else:
                ftpMODIS.retrbinary('RETR %s' % fname, open('%s/%s' % (fdirOut, fname_new), 'wb').write)
                if verbose and glob.glob('%s/%s' % (fdirOut, fname)):
                    print('Message [DOWNLOAD_MODIS]: %s has been downloaded under %s.' % (fname, fdirOut))

        except ftplib.all_errors:
            exit('Error [DOWNLOAD_MODIS]: data is not available for the requested date.')

    ftpMODIS.quit()

class MODIS_L2:

    """
    input:
        namePattern: e.g. MOD*.A20140911.2025*.hdf
        vnameExtra: default is ''
        fdir: the data directory

    output:
        a class object that contains:
            1. self.lon
            2. self.lat
            3. self.cot
            4. self.cer
            5. self.COLLOCATE(lon_in, lat_in):
                5.1. self.lon_domain
                5.2. self.lat_domain
                5.3. self.lon_collo
                5.4. self.lat_collo
                5.5. self.cot_collo
                5.6. self.cer_collo
    """

    def __init__(self, namePattern, vnameExtra='', fdir='/Users/hoch4240/Chen/mygit/MODIS-util/data/allData/6'):

        fnames = sorted(glob.glob('%s/%s' % (fdir, namePattern)))
        if len(fnames) != 2:
            exit('Error [MODIS_L2]: invalid file number for %s under %s' % (namePattern, fdir))

        self.namePattern = namePattern

        fname_geo = fnames[0]
        f_geo = SD(fname_geo, SDC.READ)
        lon = f_geo.select('Longitude')[:]
        lon[lon<0.0] += 360.0
        self.lon = lon
        self.lat = f_geo.select('Latitude')[:]
        f_geo.end()

        fname_cld = fnames[1]
        f_cld = SD(fname_cld, SDC.READ)
        vname_cot = 'Cloud_Optical_Thickness' + vnameExtra
        self.cot = f_cld.select(vname_cot)[:] * f_cld.select(vname_cot).attributes()['scale_factor']
        vname_cer = 'Cloud_Effective_Radius' + vnameExtra
        self.cer = f_cld.select(vname_cer)[:] * f_cld.select(vname_cer).attributes()['scale_factor']
        f_cld.end()

    def COLLOCATE(self, lon_in, lat_in):

        lon_in[lon_in<0.0] += 360.0
        logic = (self.lon>(lon_in.min()-0.2)) & (self.lon<(lon_in.max()+0.2)) & \
                (self.lat>(lat_in.min()-0.2)) & (self.lat<(lat_in.max()+0.2))

        self.lon_domain = self.lon[logic].ravel()
        self.lat_domain = self.lat[logic].ravel()
        self.cot_domain = self.cot[logic].ravel()
        self.cer_domain = self.cer[logic].ravel()

        points = np.array(list(zip(self.lon_domain, self.lat_domain)))

        self.lon_collo = lon_in
        self.lat_collo = lat_in
        self.cot_collo = interpolate.griddata(points, self.cot_domain, (lon_in, lat_in), method='linear')
        self.cer_collo = interpolate.griddata(points, self.cer_domain, (lon_in, lat_in), method='linear')

def WORLDVIEW_DOWNLOAD():

    import urllib.request

    rgb_link_p1 = 'https://gibs.earthdata.nasa.gov/image-download?TIME='
    rgb_link_p3 = '&extent=109.27000072667128,-0.37391741801447864,134.7231257266713,24.65733258198552&epsg=4326&layers=MODIS_'
    rgb_link_p5 = '_CorrectedReflectance_TrueColor,Coastlines,MODIS_Combined_Value_Added_AOD&opacities=1,1,1&worldfile=false&format=image/png&width=2896&height=2848'

    ctt_link_p1 = 'https://gibs.earthdata.nasa.gov/image-download?TIME='
    ctt_link_p3 = '&extent=109.27000072667128,-0.37391741801447864,134.7231257266713,24.65733258198552&epsg=4326&layers=Coastlines,MODIS_'
    ctt_link_p5 = '_Cloud_Top_Temp_Day,MODIS_Combined_Value_Added_AOD&opacities=1,1,1&worldfile=false&format=image/png&width=2896&height=2848'


    date_s = datetime.datetime(2016, 7, 1)
    date_e = datetime.datetime(2016, 9, 1)

    tags = {'Terra RGB': '01', 'Aqua RGB': '02', 'Terra CTT': '03', 'Aqua CTT': '04'}

    while date_s < date_e:

        link_p2 = date_s.strftime('%Y%j')
        date_str = date_s.strftime('%Y-%m-%d')

        for link_p4 in ['Terra', 'Aqua']:
            rgb_tag = '%s RGB' % link_p4
            ctt_tag = '%s CTT' % link_p4

            rgb_link = rgb_link_p1 + link_p2 + rgb_link_p3 + link_p4 + rgb_link_p5
            ctt_link = ctt_link_p1 + link_p2 + ctt_link_p3 + link_p4 + ctt_link_p5

            rgb_filename = '/Users/hoch4240/Chen/work/09_CAMP2Ex/data/%4.4d/%s_%s.png' % (date_s.year, date_str, tags[rgb_tag])
            ctt_filename = '/Users/hoch4240/Chen/work/09_CAMP2Ex/data/%4.4d/%s_%s.png' % (date_s.year, date_str, tags[ctt_tag])

            urllib.request.urlretrieve(rgb_link, rgb_filename)
            urllib.request.urlretrieve(ctt_link, ctt_filename)

        print(date_str)
        date_s += datetime.timedelta(days=1)


# =============================================================

class READ_ICT_HSK:

    def __init__(self, date, tmhr_range=None, fdir='/Users/hoch4240/Chen/work/01_ARISE/comp2/data/hsk'):

        date_s = date.strftime('%Y%m%d')
        self.date   = date
        self.date_s = date_s

        fnames = glob.glob('%s/*%s*' % (fdir, date_s))
        Nfiles = len(fnames)
        if Nfiles == 1:
            fname = fnames[0]
        elif Nfiles > 1:
            fname = fnames[0]
            print("Warning [READ_ICT_HSK]: found more than 1 file for %s." % date_s)
        else:
            exit('Error [READ_ICT_HSK]: no file found for %s' % date_s)

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

        if tmhr_range != None:
            tmhr0 = data[:, 0]/3600.0
            logic = (tmhr0>=tmhr_range[0]) & (tmhr0<=tmhr_range[1])
            for i, vname in enumerate(vnames):
                self.data[vname] = data[:, i][logic]
        else:
            for i, vname in enumerate(vnames):
                self.data[vname] = data[:, i]

def TEST_DOWNLOAD():

    date = datetime.datetime(2014, 9, 11)
    date_s = datetime.datetime.strftime(date, '%Y-%m-%d')

    hsk  = READ_ICT_HSK(date)
    tmhr = (hsk.data['Start_UTC']/3600.0)[::10]
    lon  = hsk.data['Longitude'][::10]
    lat  = hsk.data['Latitude'][::10]

    for satID in ['aqua', 'terra']:
        data = FIND_MODIS(date, tmhr, lon, lat, satID=satID, tmhr_range=[20.0, 23.5])
        EARTH_VIEW(data, tmhr, lon, lat)
        # ftp_init = FTP_INIT(data)
        # DOWNLOAD_MODIS(ftp_init, fdirOut='data/allData/6')

def TEST_READ():
    date = datetime.datetime(2014, 9, 11)

    # hsk  = READ_ICT_HSK(date, tmhr_range=[20.4167, 20.5])
    # namePattern = 'MOD*.A20140911.2025*.hdf'
    # modis = MODIS_L2(namePattern)
    # modis.COLLOCATE(hsk.data['Longitude'], hsk.data['Latitude'])

    # hsk  = READ_ICT_HSK(date, tmhr_range=[22.0, 22.0833])
    # namePattern = 'MOD*.A20140911.2200*.hdf'
    # modis = MODIS_L2(namePattern)
    # modis.COLLOCATE(hsk.data['Longitude'], hsk.data['Latitude'])

    # hsk  = READ_ICT_HSK(date, tmhr_range=[20.75, 20.8333])
    # namePattern = 'MYD*.A20140911.2045*.hdf'
    # modis = MODIS_L2(namePattern)
    # modis.COLLOCATE(hsk.data['Longitude'], hsk.data['Latitude'])

    # hsk  = READ_ICT_HSK(date, tmhr_range=[22.3333, 22.4167])
    # namePattern = 'MYD*.A20140911.2220*.hdf'
    # modis = MODIS_L2(namePattern)
    # modis.COLLOCATE(hsk.data['Longitude'], hsk.data['Latitude'])

    hsk  = READ_ICT_HSK(date, tmhr_range=[21.16416667, 22.60638889])
    namePattern = 'MOD*.A20140911.2025*.hdf'
    # namePattern = 'MOD*.A20140911.2200*.hdf'
    # namePattern = 'MYD*.A20140911.2045*.hdf'
    # namePattern = 'MYD*.A20140911.2220*.hdf'
    modis = MODIS_L2(namePattern)
    modis.COLLOCATE(hsk.data['Longitude'], hsk.data['Latitude'])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # proj = ccrs.PlateCarree()
    # proj_map = ccrs.LambertConformal(central_longitude=modis.lon.mean(), central_latitude=modis.lat.mean())
    # proj_map = ccrs.Stereographic(central_longitude=modis.lon.mean(), central_latitude=modis.lat.mean())

    rcParams['font.size'] = 16
    # fig = plt.figure(figsize=(5.8, 5))
    fig = plt.figure(figsize=(7, 6))
    # ax1 = fig.add_subplot(111, projection=proj_map)
    ax1 = fig.add_subplot(111)
    # ax1.set_extent([modis.lon_domain.min()-0.2, modis.lon_domain.max()+0.2, modis.lat_domain.min()-0.2, modis.lat_domain.max()+0.2], ccrs.PlateCarree())

    # ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, vmin=0.0, vmax=20.0, cmap='jet', alpha=0.2, transform=proj)
    # cs1 = ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, s=0.0, vmin=0.0, vmax=20.0, cmap='jet', alpha=1.0, transform=proj)
    # ax1.scatter(hsk.data['Longitude'], hsk.data['Latitude'], c='k', s=0.8, alpha=0.8, transform=proj)

    ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, vmin=0.0, vmax=20.0, cmap='jet', alpha=0.4, lw=0.0)
    cs1 = ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, s=0.0, vmin=0.0, vmax=20.0, cmap='jet', alpha=1.0, lw=0.0)
    ax1.scatter(modis.lon_collo, modis.lat_collo, c='k', s=1.8, alpha=0.8, lw=0.0)

    # ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cer_domain, vmin=0.0, vmax=25.0, cmap='jet', alpha=0.4, lw=0.0)
    # cs1 = ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cer_domain, s=0.0, vmin=0.0, vmax=25.0, cmap='jet', alpha=1.0, lw=0.0)
    # ax1.scatter(modis.lon_collo, modis.lat_collo, c='k', s=1.8, alpha=0.8, lw=0.0)

    # ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, vmin=0.0, vmax=20.0, cmap='jet', alpha=0.2)
    # cs1 = ax1.scatter(modis.lon_domain, modis.lat_domain, c=modis.cot_domain, s=0.0, vmin=0.0, vmax=20.0, cmap='jet', alpha=1.0)
    # ax1.scatter(modis.lon_collo, modis.lat_collo, c='k', s=0.8, alpha=0.8)
    ax1.yaxis.set_major_locator(FixedLocator(np.arange(60.0, 90.1, 0.5)))
    ax1.set_xlabel('Longitude [$^\circ$]')
    ax1.set_ylabel('Latitude [$^\circ$]')
    plt.colorbar(cs1)
    # plt.savefig('%s.png' % namePattern[:-4])
    plt.savefig('cot.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

def EARTH_VIEW_TEST(data, tmhr, lon, lat):

    """
    Purpose:
        Plot input geo info and MODIS granule on map (globe).

    input:
        data: geoMeta data

        tmhr: -
        lon : --> input geo info, e.g., flight track
        lat : -
    """

    lon[lon>180.0] -= 360.0
    logic  = (tmhr>=0.0)&(tmhr<48.0) & (lon>=-180.0)&(lon<=180.0) & (lat>=-90.0)&(lat<=90.0)

    tmhr   = tmhr[logic]
    lon    = lon[logic]
    lat    = lat[logic]

    rcParams['font.size'] = 8.0

    proj_ori = ccrs.PlateCarree()
    for i, line in enumerate(data):

        xx0  = np.array([line['GRingLongitude1'], line['GRingLongitude2'], line['GRingLongitude3'], line['GRingLongitude4'], line['GRingLongitude1']])
        yy0  = np.array([line['GRingLatitude1'] , line['GRingLatitude2'] , line['GRingLatitude3'] , line['GRingLatitude4'] , line['GRingLatitude1']])

        if (abs(xx0[0]-xx0[1])>180.0) | (abs(xx0[0]-xx0[2])>180.0) | \
           (abs(xx0[0]-xx0[3])>180.0) | (abs(xx0[1]-xx0[2])>180.0) | \
           (abs(xx0[1]-xx0[3])>180.0) | (abs(xx0[2]-xx0[3])>180.0):

            xx0[xx0<0.0] += 360.0

        xx = xx0[:-1]
        yy = yy0[:-1]
        center_lon = xx.mean()
        center_lat = yy.mean()

        # second attempt to find the center point of MODIS granule
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        proj_tmp   = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_tmp = proj_tmp.transform_points(proj_ori, xx, yy)[:, [0, 1]]
        center_xx  = LonLat_tmp[:, 0].mean(); center_yy = LonLat_tmp[:, 1].mean()
        center_lon, center_lat = proj_ori.transform_point(center_xx, center_yy, proj_tmp)
        # ---------------------------------------------------------------------

        proj_new = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
        LonLat_in = proj_new.transform_points(proj_ori, lon, lat)[:, [0, 1]]
        LonLat_modis  = proj_new.transform_points(proj_ori, xx0, yy0)[:, [0, 1]]

        ax = plt.axes(projection=proj_new)
        ax.set_global()
        ax.stock_img()
        ax.coastlines(color='gray', lw=0.2)
        title = RENAME_MODIS(data[i]['GranuleID'].decode('UTF-8'))
        date_s = title.split('.')[1][1:]
        new_date_s = '%s-%s-%s' % (date_s[:4], date_s[4:6], date_s[6:])
        ax.set_title(new_date_s, fontsize=16)

        modis_granule  = mpl_path.Path(LonLat_modis, closed=True)
        pointsIn       = modis_granule.contains_points(LonLat_in)
        percentIn      = float(pointsIn.sum()) / float(pointsIn.size) * 100.0
        if (percentIn > 0):
            patch = patches.PathPatch(modis_granule, facecolor='g', edgecolor='g', alpha=0.4, lw=0.2)
        else:
            patch = patches.PathPatch(modis_granule, facecolor='k', edgecolor='k', alpha=0.2, lw=0.2)

        cs = ax.scatter(lon[::20], lat[::20], transform=ccrs.Geodetic(), s=0.01, c='k', cmap='jet', alpha=0.5)
        # ax.scatter(xx.mean(), yy.mean(), marker='*', transform=ccrs.Geodetic(), s=6, c='r')
        # ax.scatter(center_lon, center_lat, marker='*', transform=ccrs.Geodetic(), s=6, c='b')
        ax.add_patch(patch)
        # plt.colorbar(cs, shrink=0.6)
        plt.savefig('%s.png' % '.'.join(title.split('.')[:-1]))
        plt.close()

    # ---------------------------------------------------------------------

def TEST_EARTHVIEW(date):

    date_s = datetime.datetime.strftime(date, '%Y-%m-%d')

    hsk = READ_ICT_HSK(date)
    tmhr = (hsk.data['Start_UTC']/3600.0)
    lon  = hsk.data['Longitude']
    lat  = hsk.data['Latitude']

    for satID in ['aqua', 'terra']:
        data = FIND_MODIS(date, tmhr, lon, lat, satID=satID, tmhr_range=[20.0, 23.5])
        EARTH_VIEW_TEST(data, tmhr, lon, lat)

def TEST_MODIS():
    import h5py

    namePattern = 'MOD*.A20140911.2025*.hdf'
    modis1 = MODIS_L2(namePattern)

    namePattern = 'MOD*.A20140911.2200*.hdf'
    modis2 = MODIS_L2(namePattern)

    f = h5py.File('/Users/hoch4240/Chen/work/01_ARISE/albedo/lrt_prep_20140911.h5', 'r+')
    lon = f['lon'][...]
    lat = f['lat'][...]
    tmhr = f['tmhr'][...]

    modis1.COLLOCATE(lon[tmhr<21.6], lat[tmhr<21.6])
    modis2.COLLOCATE(lon[tmhr>=21.6], lat[tmhr>=21.6])

    cot_mod = np.append(modis1.cot_collo, modis2.cot_collo)
    cer_mod = np.append(modis1.cer_collo, modis2.cer_collo)
    print(modis1.cot_collo.shape)
    print(modis2.cot_collo.shape)
    exit()

    logic = (cot_mod<0.0) | (cot_mod>1000.0) | (cer_mod<4.0) | (cer_mod>24.0)
    cot_mod[logic] = 0.0
    cer_mod[logic] = 4.0

    f['cot_mod'] = cot_mod
    f['cer_mod'] = cer_mod
    f['logic_mod'] = np.logical_not(logic)

    f.close()
    # ---------------------------------------------------------------------

if __name__ == '__main__':

    # WORLDVIEW_DOWNLOAD()
    # exit()

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import matplotlib.patches as patches
    from matplotlib.ticker import FixedLocator
    import cartopy.crs as ccrs

    TEST_READ()
    # TEST_MODIS()
    exit()

    date = datetime.datetime(2014, 9, 17)
    TEST_EARTHVIEW(date)
