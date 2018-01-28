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
            3. self.ctp: cloud thermodynamic phase
            4. self.cot
            5. self.cer
            6. self.cot_pcl
            7. self.cer_pcl
            8. self.COLLOCATE(lon_in, lat_in):
                8.1. self.lon_domain
                8.2. self.lat_domain
                8.3. self.cot_domain
                8.4. self.cer_domain
                8.5. self.cot_pcl_domain
                8.6. self.cer_pcl_domain
                8.7 . self.lon_collo
                8.8 . self.lat_collo
                8.9 . self.cot_collo
                8.10. self.cer_collo
                8.11. self.cot_pcl_collo
                8.12. self.cer_pcl_collo
                8.13. self.cot_collo_all
                8.14. self.cer_collo_all
    """

    def __init__(self, namePattern, vnameExtra='', fdir='/Users/hoch4240/Chen/mygit/MODIS-util/data/allData/6', copFlag=None):

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

        vname_ctp = 'Cloud_Phase_Optical_Properties' + vnameExtra
        self.ctp  = np.int_(f_cld.select(vname_ctp)[:] * f_cld.select(vname_ctp).attributes()['scale_factor'])

        if copFlag == None:
            vname_cot = 'Cloud_Optical_Thickness' + vnameExtra
            self.cot = f_cld.select(vname_cot)[:] * f_cld.select(vname_cot).attributes()['scale_factor']
            vname_cer = 'Cloud_Effective_Radius' + vnameExtra
            self.cer = f_cld.select(vname_cer)[:] * f_cld.select(vname_cer).attributes()['scale_factor']

            vname_cot     = 'Cloud_Optical_Thickness_PCL' + vnameExtra
            self.cot_pcl  = f_cld.select(vname_cot)[:] * f_cld.select(vname_cot).attributes()['scale_factor']
            vname_cer     = 'Cloud_Effective_Radius_PCL' + vnameExtra
            self.cer_pcl  = f_cld.select(vname_cer)[:] * f_cld.select(vname_cer).attributes()['scale_factor']

        else:
            vname_cot = 'Cloud_Optical_Thickness_%s' % (copFlag) + vnameExtra
            self.cot = f_cld.select(vname_cot)[:] * f_cld.select(vname_cot).attributes()['scale_factor']
            vname_cer = 'Cloud_Effective_Radius_%s' % (copFlag) + vnameExtra
            self.cer = f_cld.select(vname_cer)[:] * f_cld.select(vname_cer).attributes()['scale_factor']

            vname_cot     = 'Cloud_Optical_Thickness_%s_PCL' % (copFlag) + vnameExtra
            self.cot_pcl  = f_cld.select(vname_cot)[:] * f_cld.select(vname_cot).attributes()['scale_factor']
            vname_cer     = 'Cloud_Effective_Radius_%s_PCL' % (copFlag) + vnameExtra
            self.cer_pcl  = f_cld.select(vname_cer)[:] * f_cld.select(vname_cer).attributes()['scale_factor']

        f_cld.end()

    def COLLOCATE(self, lon_in, lat_in, tmhr_in=None):

        lon_in[lon_in<0.0] += 360.0
        logic = (self.lon>(lon_in.min()-0.2)) & (self.lon<(lon_in.max()+0.2)) & \
                (self.lat>(lat_in.min()-0.2)) & (self.lat<(lat_in.max()+0.2))

        self.lon_domain     = self.lon[logic].ravel()
        self.lat_domain     = self.lat[logic].ravel()
        self.cot_domain     = self.cot[logic].ravel()
        self.cer_domain     = self.cer[logic].ravel()
        self.cot_pcl_domain = self.cot_pcl[logic].ravel()
        self.cer_pcl_domain = self.cer_pcl[logic].ravel()
        self.ctp_domain     = self.ctp[logic].ravel()

        self.cot_domain_all = self.cot_domain.copy()
        self.cer_domain_all = self.cer_domain.copy()
        logic = ((self.cot_domain<0.0)&(self.cot_pcl_domain>0.0)) & ((self.cer_domain<0.0)&(self.cer_pcl_domain>0.0))
        self.cot_domain_all[logic] = self.cot_pcl_domain[logic]
        self.cer_domain_all[logic] = self.cer_pcl_domain[logic]

        points = np.array(list(zip(self.lon_domain, self.lat_domain)))

        self.lon_collo  = lon_in
        self.lat_collo  = lat_in
        if tmhr_in is not None:
            self.tmhr_collo = tmhr_in

        self.cot_collo     = interpolate.griddata(points, self.cot_domain    , (lon_in, lat_in), method='linear')
        self.cer_collo     = interpolate.griddata(points, self.cer_domain    , (lon_in, lat_in), method='linear')
        self.cot_pcl_collo = interpolate.griddata(points, self.cot_pcl_domain, (lon_in, lat_in), method='linear')
        self.cer_pcl_collo = interpolate.griddata(points, self.cer_pcl_domain, (lon_in, lat_in), method='linear')

        self.cot_collo_all = interpolate.griddata(points, self.cot_domain_all, (lon_in, lat_in), method='linear')
        self.cer_collo_all = interpolate.griddata(points, self.cer_domain_all, (lon_in, lat_in), method='linear')

        self.ctp_collo      = interpolate.griddata(points, self.ctp_domain    , (lon_in, lat_in), method='nearest')

if __name__ == '__main__':

    pass
