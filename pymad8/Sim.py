import pylab as _pl
import numpy as _np
import matplotlib.pyplot as _plt
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
    T_C.GenerateNtracks(nb_part-2)

    T_C.WriteMad8Track('../01_mad8/track_input_mad8')
    T_C.WriteBdsimTrack('../03_bdsimModel/track_input_bdsim')

    return T_C


def setSamplersAndTrack(twissfile, rmatfile, nb_sampl):
    twiss = _m8.Output(twissfile)
    rmat = _m8.Output(rmatfile, 'rmat')

    T = _m8.Sim.Tracking(twiss, rmat)
    T.GenerateSamplers(nb_sampl)
    T.AddSamplers('IP.LUXE.T20', select='name')

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

    def GenerateNtracks(self, nb):
        """
        | Add multiple track for which all parameters follow a gaussian disstribution
        """
        for i in range(nb):
            x = _pl.normal(0, 1e-6)
            xp = _pl.normal(0, 1e-6)
            y = _pl.normal(0, 1e-6)
            yp = _pl.normal(0, 1e-6)
            z = _pl.normal(0, 1e-6)
            DE = _pl.normal(0, 1e-6)
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
                self.AddSamplers(v)
            return 0
        elif select == 'index':
            if type(value) != int:
                raise ValueError("By default expect index of samplers. To give names or types use select='name' or select='type'")
            index = value
            name = self.rmat.getNamesByIndex(index)
        elif select == 'name':
            index = self.rmat.getIndexByNames(value)
            name = value
        elif select == 'type':
            index = self.rmat.getIndexByTypes(value)
            name = self.rmat.getNamesByTypes(value)
        else:
            raise ValueError("Unknown value {} for argument 'select', please use 'index', 'name' or 'type'".format(select))
        matrix = _np.reshape(self.reduced_rmat.iloc[index].tolist(), (6, 6))
        self.sampler_list[index] = {'name': name, 'matrix': matrix}

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

        if turns < 1:
            self._MakeNturns(turns)

        particle_df_dict = {}
        for sampler_index in self.sampler_list:
            sampler_name = self.sampler_list[sampler_index]['name']
            sampler_matrix = self.sampler_list[sampler_index]['matrix']
            S = self.rmat.data['S'][sampler_index]

            particle_data = []
            for track in self.initial_dict:
                initial_vector = _np.array(list(self.initial_dict[track].values()))
                final_vector = [turns, sampler_name, track, S] + list(_np.matmul(sampler_matrix, initial_vector))
                particle_data.append(final_vector)
            particle_df_dict[sampler_index] = _pd.DataFrame(particle_data, columns=['TURNS', 'SAMPLER', 'PARTICLE', 'S', 'X', 'PX', 'Y', 'PY', 'T', 'PT'])
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
        particle_df_dict = {}
        for track in self.initial_dict:
            endstr = "_part_{}_orbit.root".format(track)
            outputfile = inputfilename.replace('04_dataLocal', '06_analysis').replace('05_dataFarm', '06_analysis').replace('.root', endstr)
            _bd.Run.RebdsimOrbit(inputfilename, outputfile, index=str(track-1), silent=True)
            bdsim = _bd.Data.Load(outputfile)

            particle_data = []
            for sampler_index, sampler in enumerate(bdsim.orbit.elementName()):
                S = bdsim.orbit.s()[sampler_index]
                X = bdsim.orbit.x()[sampler_index]
                Y = bdsim.orbit.y()[sampler_index]
                XP = bdsim.orbit.xp()[sampler_index]
                YP = bdsim.orbit.yp()[sampler_index]
                vector = [sampler, track, S, X, XP, Y, YP]
                particle_data.append(vector)
            particle_df_dict[track] = _pd.DataFrame(particle_data, columns=['SAMPLER', 'PARTICLE', 'S', 'X', 'PX', 'Y', 'PY'])
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
        S = pymad8_df['S'].iloc[0]
        title = "Beam profile in {} for sampler {} (S={}m)".format(coord, sampler_name, S)

        if coord in ['X', 'Y', 'T']:
            unit = 'm'
        elif coord in ['PX', 'PY', 'PT']:
            unit = 'rad'
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))
        V_pymad8 = pymad8_df[coord].tolist()

        _plt.hist(V_pymad8, bins=15)
        _plt.xlabel("{} [{}]".format(coord, unit))
        _plt.ylabel("Number of entries")
        _plt.title(title)

    def PlotCorrelation(self, S, coord, ref_sampler='IP.LUXE.T20'):
        """
        | Correlation plot for a given coordinate at the closest sampler from given S
        | By default the reference sampler is LUXE IP
        """
        sampler_index = self.twiss.getIndexByNearestS(S)
        sampler_name = self.twiss.getNameByNearestS(S)
        sampler_S = self.twiss.data['S'][sampler_index]
        pymad8_df = self.pymad8_df.loc[self.pymad8_df['S'] == sampler_S]
        pymad8_IP_df = self.pymad8_df.loc[self.pymad8_df['SAMPLER'] == ref_sampler]
        S = pymad8_df['S'].iloc[0]
        title = "Comparison in {} to IP for sampler {} (S={}m)".format(coord, sampler_name, S)

        if coord in ['X', 'Y', 'T']:
            unit = 'm'
        elif coord in ['PX', 'PY', 'PT']:
            unit = 'rad'
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))
        V_IP_pymad8 = pymad8_IP_df[coord].tolist()
        V_pymad8 = pymad8_df[coord].tolist()

        _plt.scatter(V_pymad8, V_IP_pymad8, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("{} [{}]".format(coord, unit))
        _plt.ylabel("{}_IP [{}]".format(coord, unit))
        _plt.title(title)

    def PlotPhaseSpace(self, S, coord, bdsimCompare=False):
        """
        | Phase space plot for a given coordinate at the closest sampler from given S
        """
        sampler_index = self.twiss.getIndexByNearestS(S)
        sampler_name = self.twiss.getNameByNearestS(S)
        sampler_S = self.twiss.data['S'][sampler_index]
        pymad8_df = self.pymad8_df.loc[self.pymad8_df['S'] == sampler_S]
        S = pymad8_df['S'].iloc[0]
        title = "Phase space for sampler {} (S={}m) in {}".format(sampler_name, S, coord)

        if coord in ['X', 'Y', 'T']:
            V_pymad8 = pymad8_df[coord].tolist()
            PV_pymad8 = pymad8_df['P{}'.format(coord)].tolist()
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))

        if bdsimCompare:
            if hasattr(self, 'bdsim_df'):
                bdsim_df = self.bdsim_df[self.bdsim_df['S'] == sampler_S]
                V_bdsim = bdsim_df[coord].tolist()
                PV_bdsim = bdsim_df['P{}'.format(coord)].tolist()
                _plt.plot(V_bdsim, PV_bdsim, ls='', marker='+', label='BDSIM')
            else:
                raise AttributeError("No bdsim data found in track class")

        _plt.scatter(V_pymad8, PV_pymad8, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("{} [m]".format(coord))
        _plt.ylabel("P{} [rad]".format(coord))
        _plt.title(title)
        _plt.legend()

    def PlotTrajectory(self, particle, coord, bdsimCompare=False):
        """
        | Trajectory plot for a given coordinate and for a given particle number
        """
        pymad8_df = self.pymad8_df[self.pymad8_df['PARTICLE'] == particle]
        S_pymad8 = pymad8_df['S'].tolist()
        if coord in ['X', 'Y']:
            unit = 'm'
        elif coord in ['PX', 'PY']:
            unit = 'rad'
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))
        V_pymad8 = pymad8_df[coord].tolist()

        if bdsimCompare:
            if hasattr(self, 'bdsim_df'):
                bdsim_df = self.bdsim_df[self.bdsim_df['PARTICLE'] == particle]
                S_bdsim = bdsim_df['S'].tolist()
                V_bdsim = bdsim_df[coord].tolist()
                _plt.plot(S_bdsim, V_bdsim, ls='', marker='+', label='BDSIM')
            else:
                raise AttributeError("No bdsim data found in track class")

        _plt.scatter(S_pymad8, V_pymad8, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("S [m]")
        _plt.ylabel("{} [{}]".format(coord, unit))
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

