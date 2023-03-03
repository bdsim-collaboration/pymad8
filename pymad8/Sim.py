import pylab as _pl
import numpy as _np
import matplotlib.pyplot as _plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import pandas as _pd
import glob as _gl
import pybdsim as _bd
import pymad8 as _m8


def testTrack(twissfile, rmatfile):
    twiss = _m8.Output(twissfile)
    rmat = _m8.Output(rmatfile, 'rmat')

    T_C = _m8.Sim.Track_Collection(14)
    T_C.AddTrack(0, 0, 0, 0, 0, 0)
    T_C.AddTrack(0.01, 0, 0.01, 0, 0, 0)
    T_C.AddTrack(0, 0.01, 0, 0.01, 0, 0)
    T_C.AddTrack(0, 0, 0, 0, 0.01, 0.01)
    T_C.WriteMad8Track('TEST_track_input_mad8')
    T_C.WriteBdsimTrack('TEST_track_input_bdsim')

    T = _m8.Sim.Tracking(twiss, rmat)
    T.GenerateSamplers(10)
    T.RunPymad8Tracking(T_C)
    return T


def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def setTrackCollection(nb_part, energy, paramdict):
    T_C = _m8.Sim.Track_Collection(energy)

    T_C.AddTrack(0, 0, 0, 0, 0, 0)
    # T_C.AddTrack(0, 0, 0, 0, 0, 0.01)
    T_C.GenerateNtracks(nb_part-1, paramdict)

    T_C.WriteMad8Track('../01_mad8/track_input_mad8')
    T_C.WriteBdsimTrack('../03_bdsimModel/track_input_bdsim')

    return T_C


def setSamplersAndTrack(twissfile, rmatfile, nb_sampl):
    twiss = _m8.Output(twissfile)
    rmat = _m8.Output(rmatfile, 'rmat')

    T = _m8.Sim.Tracking(twiss, rmat)
    # T.GenerateSamplers(nb_sampl)
    # T.AddSamplers('IP.LUXE.T20', select='name')
    T.AddSamplers(['QUAD', 'RBEN', 'SBEN', 'SEXT', 'OCTU', 'MARK', 'DRIF', '    '], select='type')

    return T


def CheckUnits(coord):
    if coord in ['X', 'Y', 'T']:
        unit = 'm'
    elif coord in ['PX', 'PY', 'PT']:
        unit = 'rad'
    else:
        raise ValueError("Unknown coordinate : {}".format(coord))
    return unit


class Track_Collection:
    """
    | Class to create and store different initial particles for future tracking
    | Requires only the reference enegy in GeV
    | >>> track_collection = pymad8.Sim.Track_Collection(14)
    """
    def __init__(self, E_0):
        """
        | Stores the reference energy, the number of tracks in this collection
        | and dictionaries of track parameters in both mad8 and bdsim convention
        """
        self.ntracks = 0
        self.track_dict_mad8 = {}
        self.track_dict_bdsim = {}
        self.E_0 = E_0

    def AddTrack(self, x, xp, y, yp, z, DE):
        """
        | Add a new track to the collection with given position, angle and energy difference from the reference one
        """
        self.ntracks += 1
        self.track_dict_mad8[self.ntracks] = {'x': x, 'px': xp, 'y': y, 'py': yp, 't': z, 'deltap': DE/self.E_0}
        self.track_dict_bdsim[self.ntracks] = {'x': x, 'px': xp, 'y': y, 'py': yp, 'z': z, 'E': self.E_0 + DE}

    def GenerateNtracks(self, nb, paramdict):
        """
        | Add multiple track for which all parameters follow a gaussian disstribution
        """
        for i in range(nb):
            x = _pl.normal(paramdict['x']['mean'], paramdict['x']['std'])
            xp = _pl.normal(paramdict['xp']['mean'], paramdict['xp']['std'])
            y = _pl.normal(paramdict['y']['mean'], paramdict['y']['std'])
            yp = _pl.normal(paramdict['yp']['mean'], paramdict['yp']['std'])
            z = _pl.normal(paramdict['z']['mean'], paramdict['z']['std'])
            DE = _pl.normal(paramdict['DE']['mean'], paramdict['DE']['std'])
            self.AddTrack(x, xp, y, yp, z, DE)

    def WriteMad8Track(self, outputfile):
        """
        | Write the track collection in a file that can be read by Mad8
        """
        self.mad8_outputfile = outputfile
        f = open(outputfile, 'w')
        for part in self.track_dict_mad8:
            f.write("start")
            track = self.track_dict_mad8[part]
            for key in track:
                f.write(', ')
                if key in ['x', 'y', 't']:
                    f.write('\n')
                f.write(key)
                f.write('=')
                f.write(str(track[key]))
            f.write(";\n")
        f.close()

    def WriteBdsimTrack(self, outputfile):
        """
        | Write the track collection in a file that can be read by BDSIM
        """
        self.bdsim_outputfile = outputfile
        f = open(outputfile, 'w')
        for part in self.track_dict_bdsim:
            track = self.track_dict_bdsim[part]
            for key in track:
                f.write(str(track[key]))
                f.write(' ')
            f.write("\n")
        f.close()


class Tracking_data:
    """
    | Class that stores tracking data from Mad8 or from BDSIM
    | Comes with a bunch of useful function to extrat data
    """
    def __init__(self, dataframe):
        """
        | Stores the tracking dataframe as well as the number of particle,
        | number of samplers and the list of samplers
        """
        self.data = dataframe
        self.sampler_list = self.data[self.data['PARTICLE'] == 1]['SAMPLER'].to_list()
        self.nb_sampler = len(self.sampler_list)
        self.nb_particle = len(self.data['PARTICLE'].unique())

    def _getValues(self, data):
        """
        | Convert pandas elements of length 1 to regular value types
        | If given pandas element of length < 1 : returns a numpy array
        | Otherwise if element is already a string or integer : return unchanged element
        """
        if type(data) == str or type(data) == int:
            return data
        values_list = data.to_numpy()
        if len(values_list) == 1 and type(values_list) == list:
            return values_list[0]
        return values_list

    def getColumnsByKeys(self, keylist):
        """Return Columns that ar in keylist"""
        return self.data[keylist]

    def getRowsByValues(self, key=None, minValue=-_np.inf, maxValue=_np.inf, equalValues=None):
        """Return Rows of elements that have certain values for a chosen column"""
        if key is None:
            if equalValues is not None:
                return self.data.loc[equalValues]
            return self.data.loc[minValue:maxValue]

        column = self.getColumnsByKeys(key)
        if equalValues is not None:
            if type(equalValues) == list:
                return self.data[column.isin(equalValues)]
            return self.data[column == equalValues]
        return self.data[(column >= minValue) & (column <= maxValue)]

    def getRowsByNearestS(self, s):
        """Return Rows of the closest element in the beamline"""
        S_serie = self.data[self.data['PARTICLE'] == 1]['S'].to_numpy()
        if s < S_serie[0]:
            return self.getRowsByS(S_serie[0])
        if s >= S_serie[-1]:
            return self.getRowsByS(S_serie[-1])
        for i in range(len(S_serie)):
            if S_serie[i] <= s < S_serie[i+1]:
                if s - S_serie[i] < S_serie[i+1] - s:
                    return self.getRowsByS(S_serie[i])
                else:
                    return self.getRowsByS(S_serie[i+1])
        raise ValueError('Closest s value not found')

    def getSByNearestS(self, s):
        """Return the s position of the closest element in the beamline"""
        row = self.getRowsByNearestS(s)
        S = self._getValues(row['S'])
        if type(S) in [list, _np.ndarray]:
            return S[0]
        return S

    ####################################################################################

    def getRowsBySamplers(self, sampler):
        """Return Rows of elements with the correspondig sampler name"""
        return self.getRowsByValues(key='SAMPLER', equalValues=sampler)

    def getRowsByParticles(self, particle):
        """Return Rows of a given partcile"""
        return self.getRowsByValues(key='PARTICLE', equalValues=particle)

    def getRowsByS(self, s):
        """Return Rows of elements with the correspondig s position"""
        return self.getRowsByValues(key='S', equalValues=s)

    def getRowsBySamplerAndS(self, sampler, s):
        """Return Rows of elements with the corresponding sampler name and s position"""
        return self.data[(self.data['SAMPLER'] == sampler) & (self.data['S'] == s)]

    def getRowsBySamplerAndNearestS(self, sampler, s):
        """Return Rows of elements with the corresponding sampler name and closest s position"""
        S = self.getSByNearestS(s)
        return self.getRowsBySamplerAndS(sampler, S)

    ####################################################################################

    def getVectsByParticle(self, keys, particle):
        """Return vector of a given column and for a given particle"""
        row = self.getRowsByParticles(particle)
        return self._getValues(row[keys])

    def getVectsBySampler(self, keys, sampler):
        """Return vector of a given column and for a given sampler name"""
        row = self.getRowsBySamplers(sampler)
        return self._getValues(row[keys])

    def getVectsByS(self, keys, s):
        """Return vector of a given column and for a given s"""
        row = self.getRowsByS(s)
        return self._getValues(row[keys])

    def getVectsByNearestS(self, keys, s):
        """Return vector of a given column and for the closest s"""
        S = self.getSByNearestS(s)
        return self.getVectsByS(keys, S)

    def getVectsBySamplerAndS(self, keys, sampler, s):
        """Return vector of a given column for a given sampler name and for a given s"""
        row = self.getRowsBySamplerAndNearestS(sampler, s)
        return self._getValues(row[keys])

    def getVectsBySamplerAndNearestS(self, keys, sampler, s):
        """Return vector of a given column for a given sampler name and for the closest s"""
        S = self.getSByNearestS(s)
        return self.getVectsBySamplerAndS(keys, sampler, S)

    ####################################################################################

    def getSamplersByS(self, s):
        """Return the list of sampler name that correspond to a given s"""
        sampler = _np.unique(self.getVectsByS('SAMPLER', s))
        if len(sampler) == 1:
            return sampler[0]
        return sampler

    def getSamplersByNearestS(self, s):
        """Return the list of sampler name that correspond to the closest s"""
        S = self.getSByNearestS(s)
        return self.getSamplersByS(S)

    def getFirstSamplerByNearestS(self, s):
        """Return the first sampler name that correspond to the closest s"""
        sampler = self.getSamplersByNearestS(s)
        if type(sampler) in [list, _np.ndarray]:
            return sampler[0]
        return sampler

    def getSBySamplers(self, sampler):
        """Retrun the list of s positions that correspond to a given sampler name"""
        S = _np.unique(self.getVectsBySampler('S', sampler))
        if len(S) == 1:
            return S[0]
        return S

    ####################################################################################

    def sMin(self):
        """Return minimal S value"""
        return self.data['S'].min()

    def sMax(self):
        """Return maximal S value"""
        return self.data['S'].max()

    ####################################################################################

    def CalcCorrelChi2(self, S, coord, ref_S, ref_coord, partlimit=200):
        """Calculate the chi2, slope and intercept of the correlation fit between two positions and two coordinates"""
        sampler_name = self.getFirstSamplerByNearestS(S)
        ref_sampler_name = self.getFirstSamplerByNearestS(ref_S)

        V = self.getVectsBySamplerAndNearestS(coord, sampler_name, S)[:partlimit]
        ref_V = self.getVectsBySamplerAndNearestS(ref_coord, ref_sampler_name, ref_S)[:partlimit]

        def linear(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(linear, V, ref_V)
        slope, cst = popt
        err = sum((ref_V - slope * V - cst) ** 2)

        # cov_err = _np.sqrt(_np.diag(pcov))[0]
        return err, slope, cst

    def getChi2(self, coord, ref_S, ref_coord):
        """Return the chi2 vector along the lattice"""
        S = _np.unique(self.getVectsByParticle('S', 1))[1::2]

        Vect_Chi2 = _np.array([])
        for s in S:
            Vect_Chi2 = _np.append(Vect_Chi2, self.CalcCorrelChi2(s, coord, ref_S, ref_coord)[0])

        Curve_Chi2 = interp1d(S, Vect_Chi2, fill_value="extrapolate")
        return Curve_Chi2

    def buildBPMmatrix(self, ref_S, ref_coord, BPM_list=None, BPM_list_type='pos', s_range=[-_np.inf, _np.inf], noise=None, mean_sub=False):
        """Build the BPM matrix M with respect to a given list of BPMs"""
        ref_real_S = self.getSByNearestS(ref_S)
        ref_name = self.getFirstSamplerByNearestS(ref_S)
        ref_Vect = self.getVectsBySamplerAndNearestS(ref_coord, ref_name, ref_S)

        if BPM_list is not None:
            if BPM_list_type == 'pos':
                name_list = []
                s_list = []
                for s in BPM_list:
                    name_list.append(self.getFirstSamplerByNearestS(s))
                    s_list.append(self.getSByNearestS(s))
                reduced_df = self.data[(self.data['S'].isin(s_list)) & (self.data['SAMPLER'].isin(name_list))]
            if BPM_list_type == 'type':
                reduced_df = self.data.loc[self.data['TYPE'].isin(BPM_list)]
        else:
            reduced_df = self.data

        reduced_df = reduced_df[(reduced_df['S'] >= s_range[0]) & (reduced_df['S'] <= s_range[1])]
        reduced_df = reduced_df.drop(reduced_df.index[reduced_df['S'] == ref_real_S])

        M_X = reduced_df['X'].to_numpy().reshape((-1, self.nb_particle)).transpose()
        M_Y = reduced_df['Y'].to_numpy().reshape((-1, self.nb_particle)).transpose()
        S_Vect = reduced_df[reduced_df.index.get_level_values(1) == 0]['S'].to_numpy()
        M = _np.concatenate((M_X, M_Y), axis=1)

        if noise is not None:
            M_noise = _np.random.normal(0, noise, M.shape)
            M = M + M_noise
        if mean_sub:
            V_mean = M.mean(0)
            M = M - V_mean

        return M, ref_Vect, S_Vect

    def SVD(self, M, ref_Vect):
        """Return the correlation coefficients from a given matrix M using a Singular Value Decomposition method"""
        U, d, V_t = _np.linalg.svd(M, full_matrices=False)
        D = _np.diag(d)

        D_i = _np.linalg.inv(D)
        U_t = U.transpose()
        V = V_t.transpose()

        C = _np.dot(_np.dot(V, _np.dot(D_i, U_t)), ref_Vect)
        return C

    def CalcResolution(self, ref_coord, ref_S, BPM_list, noise=10e-6):
        """"""
        M, Real_vect, S_vect = self.buildBPMmatrix(ref_S, ref_coord, BPM_list=BPM_list, noise=noise)
        C_vect = self.SVD(M, Real_vect)

        Meas_vect = _np.dot(M, C_vect)
        Res_array = Meas_vect - Real_vect
        return Res_array

    def SortCoeff(self, ref_coord, ref_S, BPM_list, noise=10e-6):
        M, Real_vect, S_vect = self.buildBPMmatrix(ref_S, ref_coord, BPM_list=BPM_list, noise=noise)

        C_vect = self.SVD(M, Real_vect)
        C1, C2 = _np.split(C_vect, 2)
        if ref_coord in ['X', 'PX']:
            return BPM_list[_np.argsort(C1)]
        elif ref_coord in ['Y', 'PY']:
            return BPM_list[_np.argsort(C2)]
        else:
            raise ValueError('Input ref_coord should be X, Y, PX or PY')


class Tracking:
    """
    | Class to run tracking for a given track collection using Mad8 Rmat files
    | Requires the Mad8 twiss and rmat files
    | >>> tracking = pymad8.Sim.Tracking('twiss_tape', 'rmat_tape')
    |
    | Include a set of plotting function for tracjectory, phase space, ...
    | One can also load tracking data generated by BDSIM in order to make comparison plots
    """
    def __init__(self, twiss, rmat):
        """
        | Store the Mad8 twiss and rmat data as well as the number of elements
        """
        self.twiss = twiss
        self.rmat = rmat
        self.nelement = rmat.nrec

        rmat_factors_list = []
        digits = '123456'
        for i in digits:
            for j in digits:
                rmat_factors_list.append('R' + i + j)

        self.reduced_rmat = rmat.data[rmat_factors_list]
        self.sampler_list = {}

    def AddSamplers(self, value, select='index'):
        """
        | Add one or multiple samplers which will be used when runing the tracking
        | By default taking the indices of the elements we want as a samplers but one can change the select parameter to find samplers with names or types
        """
        if type(value) == list:
            for v in value:
                self.AddSamplers(v, select=select)
            return 0
        elif select == 'index':
            if type(value) != int:
                raise ValueError("By default expect index of samplers. To give names or types use select='name' or select='type'")
            index = value
            name = self.rmat.getNamesByIndex(index)
            Type = self.rmat.getTypesByIndex(index)
        elif select == 'name':
            name = value
            index = self.rmat.getIndexByNames(name)
            Type = self.rmat.getTypesByNames(name)
            if type(index) == list:
                for i in index:
                    self.AddSamplers(i)
                return 0
        elif select == 'type':
            Type = value
            index = self.rmat.getIndexByTypes(Type)
            name = self.rmat.getNamesByTypes(Type)
            if type(index) == list:
                for i in index:
                    self.AddSamplers(i)
                return 0
            else:
                self.AddSamplers(index)
        else:
            raise ValueError("Unknown value {} for argument 'select', please use 'index', 'name' or 'type'".format(select))
        matrix = _np.reshape(self.reduced_rmat.iloc[index].tolist(), (6, 6))
        self.sampler_list[index] = {'name': name, 'type': Type, 'matrix': matrix}

    def AddAllElementsAsSamplers(self):
        """
        | Register all lattice elements as samplers for the tracking
        """
        for index in range(self.nelement):
            self.AddSamplers(index)

    def GenerateSamplers(self, nb):
        """
        | Add multiple samplers spread evenly along the lattice
        """
        smax = int(self.rmat.sMax())
        for i in range(nb):
            s = smax*i/nb
            self.AddSamplers(self.rmat.getIndexByNearestS(s))

    def _MakeNturns(self, turns):
        """
        | In the case of circular machine, propagate the particles through multiple turns
        """
        # get the last element
        index = self.rmat.nrec-1
        matrix = _np.reshape(self.reduced_rmat.iloc[index].tolist(), (6, 6))
        for turn in range(turns):
            for track in self.initial_dict:
                initial_vector = _np.array(list(self.initial_dict[track].values()))
                final_vector = _np.matmul(matrix, initial_vector)
                for i, param in enumerate(self.initial_dict[track]):
                    self.initial_dict[track][param] = final_vector[i]

    def RunPymad8Tracking(self, track_collection, turns=1):
        """
        | Run tracking for all particles using the rmat from Mad8
        | >>> track.RunPymad8Tracking(track_collection)
        """
        self.initial_dict = track_collection.track_dict_mad8
        n_part = len(self.initial_dict)
        n_sampler = len(self.sampler_list)
        print('Mad8.tracking > {} particles and {} samplers'.format(n_part, n_sampler))

        if turns < 1:
            self._MakeNturns(turns)

        particle_df_dict = {}
        self.nb_mad8_sampler = 0
        for sampler_index in self.sampler_list:
            self.nb_mad8_sampler += 1
            sampler_name = self.sampler_list[sampler_index]['name']
            sampler_type = self.sampler_list[sampler_index]['type']
            sampler_matrix = self.sampler_list[sampler_index]['matrix']
            S = self.rmat.data['S'][sampler_index]

            particle_data = []
            for track in self.initial_dict:
                initial_vector = _np.array(list(self.initial_dict[track].values()))
                final_vector = [turns, sampler_name, sampler_type, track, S] + list(_np.matmul(sampler_matrix, initial_vector))
                particle_data.append(final_vector)
            particle_df_dict[sampler_index] = _pd.DataFrame(particle_data, columns=['TURNS', 'SAMPLER', 'TYPE', 'PARTICLE', 'S', 'X', 'PX', 'Y', 'PY', 'T', 'PT'])
        self.pymad8_df = _pd.concat(particle_df_dict, axis=0)

    def LoadMad8Track(self, inputfilename):
        """
        | Load the Mad8 track files and store it in a structure similar to the pymad8 generated data
        | /!/ NOT WORKING /!/
        """
        filelist = _gl.glob(inputfilename)
        particle_df_dict = {}
        for file in filelist:
            sampler = _m8.Output(file, 'track')

    def LoadBdsimTrack(self, inputfilename):
        """
        | Load Bdsim root file and generate orbit files for each particle in the track collection
        | Then store the data in a structure similar to the pymad8 generated one
        """
        bdsim = _bd.Data.Load(inputfilename)
        samplers = bdsim.GetSamplerNames()
        self.nb_bdsim_sampler = samplers.size()
        self.bdsim_sampler_list = {}
        particle_df_dict = {}
        for sampler_index, sampler_name in enumerate(samplers):
            _printProgressBar(sampler_index, samplers.size(),
                              prefix='Loading file {}. Track {} particles at {} samplers:'.format(inputfilename, len(self.initial_dict), samplers.size()),
                              suffix='Complete', length=50)
            self.bdsim_sampler_list[sampler_index] = {'name': sampler_name}
            sampler_data = _bd.Data.SamplerData(bdsim, samplerIndexOrName=samplers[sampler_index])
            S = sampler_data.data['S']
            X = sampler_data.data['x']
            Y = sampler_data.data['y']
            XP = sampler_data.data['xp']
            YP = sampler_data.data['yp']
            if len(self.initial_dict) != len(S):
                raise ValueError("Inconsisstant number of particles between Mad8 and Bdsim")

            particle_data = []
            for i_track, track in enumerate(self.initial_dict):
                s = S[i_track]
                x = X[i_track]
                y = Y[i_track]
                xp = XP[i_track]
                yp = YP[i_track]
                vector = [sampler_name, track, s, x, xp, y, yp]
                particle_data.append(vector)
            particle_df_dict[sampler_index] = _pd.DataFrame(particle_data, columns=['SAMPLER', 'PARTICLE', 'S', 'X', 'PX', 'Y', 'PY'])
        _printProgressBar(samplers.size(), samplers.size(),
                          prefix='Loading file {}. Track {} particles at {} samplers:'.format(inputfilename, len(self.initial_dict), samplers.size()),
                          suffix='Complete', length=50)
        self.bdsim_df = _pd.concat(particle_df_dict, axis=0)

    #########
    # PLOTS #
    #########
    def PlotHist(self, S, coord):
        """
        | Plot histogram for a given coordinate at the closest sampler from given S
        """
        sampler_index = self.twiss.getIndexByNearestS(S)
        sampler_name = self.twiss.getNameByNearestS(S)
        sampler_S = self.twiss.data['S'][sampler_index]
        pymad8_df = self.pymad8_df.loc[self.pymad8_df['S'] == sampler_S]
        title = "Beam profile in {} for sampler {} (S={}m)".format(coord, sampler_name, sampler_S)

        unit = CheckUnits(coord)
        V_pymad8 = pymad8_df[coord].tolist()

        _plt.hist(V_pymad8, bins=15)
        _plt.xlabel("{} [{}]".format(coord, unit))
        _plt.ylabel("Number of entries")
        _plt.title(title)

    def PlotCorrelation(self, index, coord, ref_index, ref_coord, linFit=False, noPlots=False):
        """
        | Correlation plot for a given coordinate at the closest sampler from given S
        | By default the reference sampler is LUXE IP
        """
        sampler_name, sampler_S, sampler_df = self.reduceDFbyIndex(index)
        ref_sampler_name, ref_sampler_S, ref_sampler_df = self.reduceDFbyIndex(ref_index)

        title = "Correlation between :\n{} at {} (S~{} m)  &  {} at {} (S~{} m)".format(coord, sampler_name, round(sampler_S, 2),
                                                                                  ref_coord, ref_sampler_name, round(ref_sampler_S, 2))

        unit = CheckUnits(coord)
        ref_unit = CheckUnits(ref_coord)

        V = sampler_df[coord].tolist()
        V_ref = ref_sampler_df[ref_coord].tolist()

        if linFit:
            def linear(x, a, b):
                return a * x + b
            popt, pcov = curve_fit(linear, V, V_ref)
            slope = popt[0]
            cst = popt[1]
            err = 0
            for x, y in zip(V, V_ref):
                err += (y - slope*x - cst)**2

            cov_err = _np.sqrt(_np.diag(pcov))[0]
            if noPlots:
                return slope, err, cov_err
            _plt.plot(V, linear(_np.array(V), *popt), color='green', label='Fit: {}_ref = {} * {} + {}'.format(ref_coord, slope, coord, cst))

        _plt.scatter(V, V_ref, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("{} [{}]".format(coord, unit))
        _plt.ylabel("{}_ref [{}]".format(ref_coord, ref_unit))
        _plt.title(title)
        _plt.legend()

    def PlotPhaseSpace(self, S, coord, linFit=False):
        """
        | Phase space plot for a given coordinate at the closest sampler from given S
        """
        sampler_index = self.twiss.getIndexByNearestS(S)
        sampler_name = self.twiss.getNameByNearestS(S)
        sampler_S = self.twiss.data['S'][sampler_index]
        pymad8_df = self.pymad8_df.loc[self.pymad8_df['S'] == sampler_S]
        title = "Phase space in {} for sampler {} (S={}m)".format(coord, sampler_name, sampler_S)

        if coord in ['X', 'Y', 'T']:
            V_pymad8 = pymad8_df[coord].tolist()
            PV_pymad8 = pymad8_df['P{}'.format(coord)].tolist()
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))

        if linFit:
            slope, b = _np.polyfit(V_pymad8, PV_pymad8, 1)
            _plt.plot(V_pymad8, slope * _np.array(V_pymad8) + b, color='green', label='Fit: slope = {}'.format(slope))

        _plt.scatter(V_pymad8, PV_pymad8, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("{} [m]".format(coord))
        _plt.ylabel("P{} [rad]".format(coord))
        _plt.title(title)
        _plt.legend()

    def PlotTrajectory(self, particle, coord, bdsimCompare=False, relativePlots=False):
        """
        | Trajectory plot for a given coordinate and for a given particle number
        """
        pymad8_df = self.pymad8_df[self.pymad8_df['PARTICLE'] == particle]
        pymad8_df = pymad8_df.sort_values('S')
        S_pymad8 = pymad8_df['S'].tolist()

        unit  = CheckUnits(coord)
        V_pymad8 = pymad8_df[coord].tolist()

        if bdsimCompare:
            if hasattr(self, 'bdsim_df'):
                bdsim_df = self.bdsim_df[self.bdsim_df['PARTICLE'] == particle]
                bdsim_df = bdsim_df.sort_values('S')
                S_bdsim = bdsim_df['S'].tolist()
                V_bdsim = bdsim_df[coord].tolist()
            else:
                raise AttributeError("No bdsim data found in track class")

        if not relativePlots:
            if bdsimCompare:
                _plt.scatter(S_pymad8, V_pymad8, facecolors='none', edgecolors='r', label='Mad8')
                _plt.plot(S_bdsim, V_bdsim, ls='', marker='+', label='BDSIM')
            else:
                _plt.plot(S_pymad8, V_pymad8, label='Mad8 particle {}'.format(particle))
            _plt.ylabel("{} [{}]".format(coord, unit))

        if relativePlots and bdsimCompare:
            f_pymad8 = interp1d(S_pymad8, V_pymad8, fill_value="extrapolate")
            f_bdsim = interp1d(S_bdsim, V_bdsim, fill_value="extrapolate")
            if coord in ['X', 'Y']:
                _plt.ylabel("X/Y [m]")
            if coord in ['PX', 'PY']:
                _plt.ylabel("PX/PY [rad]")

            def f_relat(s):
                return f_bdsim(s) - f_pymad8(s)

            V_relat = f_relat(S_bdsim)
            _plt.plot(S_bdsim, V_relat, ls='-', marker='', label='BDSIM-Mad8 in {} : std = {}'.format(coord, _np.std(V_relat)))

        _plt.xlabel("S [m]")
        _plt.legend()

    def reduceDFbyIndex(self, index):
        row = self.twiss.getRowsByIndex(index)
        name = row['NAME']
        S = row['S']
        reduced_df = self.pymad8_df.loc[self.pymad8_df['S'] == S]
        reduced_df = reduced_df[reduced_df['SAMPLER'] == name]
        if reduced_df.empty:
            raise ValueError("Element of index {} is not registered as a sampler".format(index))
        return name, S, reduced_df


# BELOW IS OLD
# ============================================

def _testTrackOld(rmatFile, nparticle=10):
    r = _m8.Output(rmatFile, "rmat")
    r.subline('IEX', -1)
    s = _Track(r)
    s.trackParticles(nparticle)
    return s


class _Track:
    def __init__(self, rmat):
        self.rmat       = rmat
        self.nelement   = rmat.nrec
        # Turn rmat data into matrix
        self.rmatMatrix = _np.reshape(rmat.data[:, 0:36], (self.nelement, 6, 6))

    def trackParticles(self, nparticle):
        # store the number of particles tracked 
        self.nparticle = nparticle
        
        # Nparticle, Nelement phase space storage
        self.psVector = _np.zeros((nparticle, self.nelement, 6))

        # loop over particles
        for i in range(0, nparticle):
            print('Track.Simple.track> particle ', i)
            p = self.generate() 
            psv = self.trackParticle(p)
            self.psVector[i] = psv
    
    def generate(self):
        return _np.array([0.0,
                         0.0,
                         _pl.normal(0, 1e-6),
                         0.0,
                         0.0,
                         0.0], dtype=_np.double)

    def trackParticle(self, p):
        # Tracking vector 
        psVector   = _np.zeros((self.rmatMatrix.shape[0], 6))

        # Set first element
        i = 1 
        psVector[0] = p 
                
        # loop over all elements from start to end 
        for m in self.rmatMatrix[1:]:
            # psVector[i] = np.dot(m,psVector[i-1])
            psVector[i] = _np.dot(m, p)
            i = i+1

        return psVector

