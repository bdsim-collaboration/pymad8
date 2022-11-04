import pylab as _pl
import numpy as _np
import matplotlib.pyplot as _plt
import pandas as _pd
import glob as _gl
import pymad8 as _m8


def testTrack(twissfile, rmatfile, nb_part, nb_sampl):
    twiss = _m8.Output(twissfile)
    rmat = _m8.Output(rmatfile, 'rmat')

    T_C = _m8.Sim.Track_Collection(14)
    T_C.AddTrack(0, 0, 0, 0, 0, 0)
    # T_C.AddTrack(0.001, 0, 0, 0, 0, 0)
    # T_C.AddTrack(0, 0.001, 0, 0, 0, 0)
    # T_C.AddTrack(0, 0, 0.001, 0, 0, 0)
    # T_C.AddTrack(0, 0, 0, 0.001, 0, 0)
    # T_C.AddTrack(0, 0, 0, 0, 0.001, 0)
    # T_C.AddTrack(0, 0, 0, 0, 0, 0.001)
    # T_C.AddTrack(0.001, 0.001, 0, 0, 0, 0)
    # T_C.AddTrack(0, 0, 0.001, 0.001, 0, 0)
    # T_C.AddTrack(0, 0, 0, 0, 0.001, 0.001)
    T_C.GenerateNtracks(nb_part)
    T_C.WriteMad8Track('../01_mad8/TEST_track_input_mad8')
    T_C.WriteBDSIMTrack('../03_bdsimModel/TEST_track_input_bdsim')

    T = _m8.Sim.Tracking(twiss, rmat)
    T.GenerateSamplers(nb_sampl)
    T.RunPymad8Tracking(T_C)
    return T


class Track_Collection:
    def __init__(self, E_0):
        self.ntracks = 0
        self.track_dict_mad8 = {}
        self.track_dict_bdsim = {}
        self.E_0 = E_0

    def AddTrack(self, x, xp, y, yp, z, DE):
        self.ntracks += 1
        self.track_dict_mad8[self.ntracks] = {'x': x, 'px': xp, 'y': y, 'py': yp, 't': z, 'deltap': DE/self.E_0}
        self.track_dict_bdsim[self.ntracks] = {'x': x, 'px': xp, 'y': y, 'py': yp, 'z': z, 'E': self.E_0 + DE}

    def GenerateNtracks(self, nb):
        for i in range(nb):
            x = _pl.normal(0, 1e-6)
            xp = _pl.normal(0, 1e-6)
            y = _pl.normal(0, 1e-6)
            yp = _pl.normal(0, 1e-6)
            z = _pl.normal(0, 1e-6)
            DE = -0.007
            self.AddTrack(x, xp, y, yp, z, DE)

    def WriteMad8Track(self, outputfile):
        f = open(outputfile, 'w')
        for part in self.track_dict_mad8:
            f.write("start")
            track = self.track_dict_mad8[part]
            for key in track:
                f.write(' ,')
                f.write(key)
                f.write('=')
                f.write(str(track[key]))
            f.write(";\n")
        f.close()

    def WriteBDSIMTrack(self, outputfile):
        f = open(outputfile, 'w')
        for part in self.track_dict_bdsim:
            track = self.track_dict_bdsim[part]
            for key in track:
                f.write(str(track[key]))
                f.write(' ')
            f.write("\n")
        f.close()


class Tracking:
    def __init__(self, twiss, rmat):
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
        smax = int(self.rmat.sMax())
        for i in range(nb):
            s = smax*i/nb
            self.AddSamplers(self.rmat.getIndexByNearestS(s))

    def MakeNturns(self, turns):
        # get the last element
        index = self.rmat.nrec-1
        matrix = _np.reshape(self.reduced_rmat.iloc[index].tolist(), (6, 6))
        for turn in range(turns):
            for track in self.initial_dict:
                initial_vector = _np.array(list(self.initial_dict[track].values()))
                final_vector = _np.matmul(initial_vector, matrix)
                for i, param in enumerate(self.initial_dict[track]):
                    self.initial_dict[track][param] = final_vector[i]

    def RunPymad8Tracking(self, track_collection, turns=1):
        self.initial_dict = track_collection.track_dict_mad8

        if turns < 1:
            self.MakeNturns(turns)

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
        filelist = _gl.glob(inputfilename)
        particle_df_dict = {}
        for file in filelist:
            pass

    def LoadBDSIMtrack(self):
        pass

    def PlotPhaseSpace(self, particle=None, sampler=None):
        if particle and sampler:
            raise SyntaxError("Could not use 'particle' and 'sampler' at the same time")
        if particle is not None:
            sub_df = self.pymad8_df[self.pymad8_df['PARTICLE'] == particle]
            title = "Phase space for particle {}".format(particle)
        elif sampler is not None:
            sub_df = self.pymad8_df.loc[self.pymad8_df['SAMPLER'] == sampler]
            S = sub_df['S'].iloc[0]
            title = "Phase space for sampler {} (S={})".format(sampler, S)
        else:
            raise ValueError("Must give one of the two arguments 'particle' or 'sampler'")
        X = sub_df['X'].tolist()
        PX = sub_df['PX'].tolist()
        Y = sub_df['Y'].tolist()
        PY = sub_df['PY'].tolist()

        _plt.subplot(1, 2, 1)
        _plt.plot(X, PX, ls='', marker='.')
        _plt.xlabel("X [m]")
        _plt.ylabel("PX [rad]")
        _plt.title("{} in X".format(title))

        _plt.subplot(1, 2, 2)
        _plt.plot(Y, PY, ls='', marker='.')
        _plt.xlabel("Y [m]")
        _plt.ylabel("PY [rad]")
        _plt.title("{} in Y".format(title))

    def PlotTrackAlongS(self, particle, coord):
        sub_df = self.pymad8_df[self.pymad8_df['PARTICLE'] == particle]
        S = sub_df['S'].tolist()
        if coord in ['X', 'Y']:
            unit = 'm'
        elif coord in ['PX', 'PY']:
            unit = 'rad'
        else:
            raise ValueError("Unknown coordinate : {}".format(coord))
        V = sub_df[coord].tolist()

        _plt.scatter(S, V, facecolors='none', edgecolors='r', label='Mad8')
        _plt.xlabel("S [m]")
        _plt.ylabel("{} [{}]".format(coord, unit))
        _plt.legend()


# BELOW IS OLD
# ============================================

def testTrackOld(rmatFile, nparticle=10):
    r = _m8.Output(rmatFile, "rmat")
    r.subline('IEX', -1)
    s = Track(r)
    s.trackParticles(nparticle)
    return s


class Track:
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

