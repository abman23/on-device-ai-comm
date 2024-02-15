from sionna.channel import AWGN, FlatFadingChannel
from sionna.fec.polar import Polar5GEncoder, Polar5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.mapping import Mapper, Demapper, Constellation

from sionna.mimo import mf_equalizer, StreamManagement, lmmse_equalizer
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna,  UMi, UMa, RMa
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel

from sionna.utils import ebnodb2no, expand_to_rank, QAMSource

import numpy as np
import tensorflow as tf

from transformers.utils import (
    logging,
)
logger = logging.get_logger("transformers")

class AWGNModel(tf.keras.Model):
    """
    Configure AWGN Channel components.
    ref: https://nvlabs.github.io/sionna/api/channel.wireless.html?highlight=awgn#sionna.channel.AWGN

    Parameters
    ----------
        :param fec_type: str, One of ["Polar5G", "LDPC5G"]
        :param num_bits_per_symbol: int
        :param fec_n: int
        :param fec_k: int 
        :param ebno_db: float
        :param ebno_db_min: float
        :param ebno_db_max: float
        :param fec_num_iter: int

    """
    def __init__(self,
                 fec_type,
                 num_bits_per_symbol,
                 fec_n,
                 fec_k,
                 ebno_db=None,
                 ebno_db_min=None,
                 ebno_db_max=None,
                 fec_num_iter=6
                 ):
        super().__init__()
        self.fec_type = fec_type
        
        self._n = fec_n
        self._k = fec_k
        self._coderate = self._k / self._n
        
        print(f'{self._k=}')
        print(f'{self._n=}')
        print(f'{self._coderate=}')

        constellation = Constellation("qam",
                                    num_bits_per_symbol,
                                    trainable=False)
        logger.info(f'Constellation: type={constellation._constellation_type} ' + \
                    f'{num_bits_per_symbol=} trainable={constellation._trainable}')
        self.num_bits_per_symbol = num_bits_per_symbol
        self.mapper = Mapper(constellation=constellation)
        
        self.channel = AWGN()

        # channel noise
        assert ebno_db is not None or (ebno_db_min is not None and ebno_db_max is not None), "Set a single ebno_db or (ebno_db_min and ebno_db_max)"
        if ebno_db is not None:
            self.ebno_db = float(ebno_db)
        else:
            self.ebno_db = ebno_db # None
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max

        print(f'{self.ebno_db=}')
        print(f'{self.ebno_db_min=}')
        print(f'{self.ebno_db_max=}')

        self.demapper = Demapper("app", constellation=constellation)
        
        # FEC
        if self.fec_type == 'Polar5G':
            self._encoder = Polar5GEncoder(self._k, self._n)
            self._decoder = Polar5GDecoder(
                self._encoder,
                dec_type='SC',
                list_size=8
            )
        elif self.fec_type == 'LDPC5G':
            self.fec_num_iter = fec_num_iter
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True, num_iter=self.fec_num_iter)
        else:
            raise ValueError(f"Invalid channel coding type: {fec_type}")
    
        
    @tf.function
    def call(self, input):
        '''
        Input
        -----
            :param input: 
        
        Output
        ------
            :return b_hat: 
        '''
        # reshape input
        input_shape = input.shape
        divisor=self._k
        if np.prod(input_shape) % divisor != 0:
            flatten_input = tf.reshape(input, [-1])
            flatten_input_len = len(flatten_input)
            
            dummy_cnt = ((flatten_input_len // divisor)+1) * divisor - flatten_input_len
            flatten_input = tf.concat([flatten_input, [0 for _ in range(dummy_cnt)]],0)
        else:
            flatten_input = input
        
        # Channel encoder        
        b = tf.reshape(flatten_input, (-1, self._k))
        codewords = self._encoder(b)
        
        # Modulation
        x = self.mapper(codewords)

        #####################
        # Channel
        #####################
        # Sampling a batch of SNRs
        batch_size=b.shape[0]
        if self.ebno_db_min is not None and self.ebno_db_max is not None:
            ebno_db_tf = tf.random.uniform(shape=[batch_size], minval=self.ebno_db_min, maxval=self.ebno_db_max)
            no = ebnodb2no(ebno_db_tf, self.num_bits_per_symbol, self._coderate)
        else:
            no = ebnodb2no(self.ebno_db, self.num_bits_per_symbol, self._coderate)

        no = expand_to_rank(no, 2)

        y = self.channel([x, no])

        #####################
        # Receiver
        #####################
        # Demodulation
        llr = self.demapper([y, no])
        llr = tf.reshape(llr, (-1, self._n))

        # Channel decoder
        b_hat = self._decoder(llr)

        if np.prod(input_shape) % divisor != 0:
            #Reshape b_hat to the original shape by cutting the arbitrarily appended elements
            flatten_b_hat = tf.reshape(b_hat, [-1])
            sliced_b_hat = flatten_b_hat[:-dummy_cnt]
            b_hat=tf.reshape(sliced_b_hat, input_shape)
        else:
            b_hat=tf.reshape(b_hat, input_shape)

        return b_hat

class CDLModel(tf.keras.Model):
    """
    Configure CDL Channel components.

    Parameters
    ----------
        :param fec_type: str, One of ["Polar5G", "LDPC5G"]
        :param cdl_model: str, One of ["A", "B", "C", "D", "E"]
        :param channel_num_tx_ant: int
        :param channel_num_rx_ant: int
        :param num_bits_per_symbol: int
        :param ebno_db: float
        :param ebno_db_min: float
        :param ebno_db_max: float
        :param fec_num_iter: int
    """
    def __init__(self,
                 fec_type,
                 cdl_model,
                 channel_num_tx_ant,
                 channel_num_rx_ant,
                 num_bits_per_symbol,
                 ebno_db=None,
                 ebno_db_min=None,
                 ebno_db_max=None,
                 fec_num_iter=6
                 ):
        super().__init__()

        # Provided parameters
        DL_CONFIG={
            "cdl_model" : cdl_model,
            "delay_spread" : 100e-9,
            "domain" : "time",
            "direction" : "downlink",
            "perfect_csi" : True,
            "speed" : 0.0,
            "cyclic_prefix_length" : 6,
            "pilot_ofdm_symbol_indices" : [2, 11],
            "duration" : None
        }
        self._domain = DL_CONFIG["domain"]
        self._direction = DL_CONFIG["direction"]
        self._cdl_model = DL_CONFIG["cdl_model"]
        self._delay_spread = DL_CONFIG["delay_spread"]
        self._perfect_csi = DL_CONFIG["perfect_csi"]
        self._speed = DL_CONFIG["speed"]
        self._cyclic_prefix_length = DL_CONFIG["cyclic_prefix_length"]
        self._pilot_ofdm_symbol_indices = DL_CONFIG["pilot_ofdm_symbol_indices"]

        logger.info(f'{DL_CONFIG=}')

        # System parameters
        self._carrier_frequency = 2.6e9
        self._subcarrier_spacing = 15e3 #subcarrier_spacing
        self._fft_size = 36 #72
        self._num_ofdm_symbols = 12 #14
        self._num_ut_ant = int(channel_num_tx_ant) #2 # Must be a multiple of two as dual-polarized antennas are used
        self._num_bs_ant = int(channel_num_rx_ant) #2 # Must be a multiple of two as dual-polarized antennas are used
        self._num_streams_per_tx = self._num_ut_ant
        self._dc_null = True
        self._num_guard_carriers = [5, 6]
        self._pilot_pattern = "kronecker"
        self._pilot_ofdm_symbol_indices = DL_CONFIG["pilot_ofdm_symbol_indices"]
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._coderate = 0.5

        # Required system components
        self._sm = StreamManagement(np.array([[1]]), self._num_streams_per_tx)

        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing = self._subcarrier_spacing,
                                num_tx=1,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern=self._pilot_pattern,
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # log
        print(f'{self._rg.num_data_symbols=}')
        # self._rg.num_data_symbols = self.num_effective_subcarriers * self._num_ofdm_symbols - \
        #       self.num_pilot_symbols
        print(f'{self._rg.num_effective_subcarriers=}')
        print(f'{self._rg._num_ofdm_symbols=}')
        print(f'{self._rg.num_pilot_symbols=}')
        # self._rg.num_effective_subcarriers= self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        print(f'{self._rg._fft_size=}')
        print(f'{self._rg._dc_null=}')
        print(f'{np.sum(self._rg._num_guard_carriers)=}')

        self._n = int(self._rg.num_data_symbols * self._num_bits_per_symbol)
        self._k = int(self._n * self._coderate)

        self._ut_array = AntennaArray(num_rows=1,
                                    num_cols=int(self._num_ut_ant/2),
                                    polarization="dual",
                                    polarization_type="cross",
                                    antenna_pattern="38.901",
                                    carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                    num_cols=int(self._num_bs_ant/2),
                                    polarization="dual",
                                    polarization_type="cross",
                                    antenna_pattern="38.901",
                                    carrier_frequency=self._carrier_frequency)

        self._cdl = CDL(model=self._cdl_model,
                        delay_spread=self._delay_spread,
                        carrier_frequency=self._carrier_frequency,
                        ut_array=self._ut_array,
                        bs_array=self._bs_array,
                        direction=self._direction,
                        min_speed=self._speed)

        self._frequencies = subcarrier_frequencies(self._rg.fft_size, self._rg.subcarrier_spacing)

        if self._domain == "freq":
            self._channel_freq = ApplyOFDMChannel(add_awgn=True)

        elif self._domain == "time":
            self._l_min, self._l_max = time_lag_discrete_time_channel(self._rg.bandwidth)
            self._l_tot = self._l_max - self._l_min + 1
            self._channel_time = ApplyTimeChannel(self._rg.num_time_samples,
                                                l_tot=self._l_tot,
                                                add_awgn=True)
            self._modulator = OFDMModulator(self._cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(self._fft_size, self._l_min, self._cyclic_prefix_length)

        self.fec_type = fec_type

        # channel noise
        assert ebno_db is not None or (ebno_db_min is not None and ebno_db_max is not None), "Set a single ebno_db or (ebno_db_min and ebno_db_max)"
        if ebno_db is not None:
            self.ebno_db = float(ebno_db)
        else:
            self.ebno_db = ebno_db # None
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max

        logger.info(f'{self.ebno_db=}')
        logger.info(f'{self.ebno_db_min=}')
        logger.info(f'{self.ebno_db_max=}')

        print(f'{self._k=}')
        print(f'{self._n=}')
        print(f'{self._coderate=}')

        print(f'{self._fft_size=}')
        print(f'{self._num_ofdm_symbols=}')
        print(f'{self._num_bits_per_symbol=}')

        print(f'{self._cdl_model=}')
        print(f'{self.fec_type=}')
        print(f'{self._num_ut_ant=}')
        print(f'{self._num_bs_ant=}')

        # FEC
        if self.fec_type == 'Polar5G':
            self._encoder = Polar5GEncoder(self._k, self._n)
            self._decoder = Polar5GDecoder(
                self._encoder,
                dec_type='SC',
                list_size=8
            )
        elif self.fec_type == 'LDPC5G':
            self.fec_num_iter = fec_num_iter
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True, num_iter=self.fec_num_iter)
        else:
            raise ValueError(f"Invalid channel coding type: {fec_type}")
        
        print(f'{self.fec_num_iter=}')

        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        if self._direction == "downlink":
            self._zf_precoder = ZFPrecoder(self._rg, self._sm, return_effective_channel=True)

        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)
        
        self._remove_nulled_scs = RemoveNulledSubcarriers(self._rg)

    @tf.function(jit_compile=True)
    def call(self, input):
        """
        Input
        -----
            :param input: 
        
        Output
        ------
            :return b_hat: 
        """
        # reshape input
        input_shape = input.shape

        divisor=self._num_streams_per_tx * self._k
        if np.prod(input_shape) % divisor != 0:
            flatten_input = tf.reshape(input, [-1])
            flatten_input_len = len(flatten_input)
            
            dummy_cnt = ((flatten_input_len // divisor)+1) * divisor - flatten_input_len
            flatten_input = tf.concat([flatten_input, [0 for _ in range(dummy_cnt)]],0)
        else:
            flatten_input = input

        b = tf.reshape(flatten_input, (-1, 1, self._num_streams_per_tx, self._k))
        batch_size = b.shape[0]

        if self.ebno_db_min is not None and self.ebno_db_max is not None:
            ebno_db_tf = tf.random.uniform(shape=[batch_size], minval=self.ebno_db_min, maxval=self.ebno_db_max)
            no = ebnodb2no(ebno_db_tf, self._num_bits_per_symbol, self._coderate, self._rg)
        else:
            no = ebnodb2no(self.ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)

        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)

        if self._domain == "time":
            # Time-domain simulations
            a, tau = self._cdl(batch_size, self._rg.num_time_samples+self._l_tot-1, self._rg.bandwidth)
            h_time = cir_to_time_channel(self._rg.bandwidth, a, tau,
                                         l_min=self._l_min, l_max=self._l_max, normalize=True)

            # As precoding is done in the frequency domain, we need to downsample
            # the path gains `a` to the OFDM symbol rate prior to converting the CIR
            # to the channel frequency response.
            a_freq = a[...,self._rg.cyclic_prefix_length:-1:(self._rg.fft_size+self._rg.cyclic_prefix_length)]
            a_freq = a_freq[...,:self._rg.num_ofdm_symbols]
            h_freq = cir_to_ofdm_channel(self._frequencies, a_freq, tau, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder([x_rg, h_freq])

            x_time = self._modulator(x_rg)
            y_time = self._channel_time([x_time, h_time, no])

            y = self._demodulator(y_time)

        elif self._domain == "freq":
            # Frequency-domain simulations

            cir = self._cdl(batch_size, self._rg.num_ofdm_symbols, 1/self._rg.ofdm_symbol_duration)
            h_freq = cir_to_ofdm_channel(self._frequencies, *cir, normalize=True)

            if self._direction == "downlink":
                x_rg, g = self._zf_precoder([x_rg, h_freq])

            y = self._channel_freq([x_rg, h_freq, no])

        if self._perfect_csi:
            if self._direction == "uplink":
                h_hat = self._remove_nulled_scs(h_freq)
            elif self._direction =="downlink":
                h_hat = g
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est ([y, no])

        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)

        if np.prod(input_shape) % divisor != 0:
            #Reshape b_hat to the original shape by cutting the arbitrarily appended elements
            flatten_b_hat = tf.reshape(b_hat, [-1])
            sliced_b_hat = flatten_b_hat[:-dummy_cnt]
            b_hat=tf.reshape(sliced_b_hat, input_shape)
        else:
            b_hat=tf.reshape(b_hat, input_shape)

        return b_hat

class RealisticModel(tf.keras.Model):
    """
    OFDM MIMO transmissions over a 3GPP 38.901 model.

    Parameters
    ----------
        :param fec_type: str, One of ["Polar5G", "LDPC5G"]
        :param scenario: str, One of ["umi", "uma", "rma"]
        :param perfect_csi: boolean
        :param channel_num_tx_ant: int
        :param channel_num_rx_ant: int
        :param num_bits_per_symbol: int
        :param ebno_db: float
        :param ebno_db_min: float
        :param ebno_db_max: float
        :param fec_num_iter: int
    """
    def __init__(self,
                 fec_type,
                 scenario,
                 perfect_csi,
                 channel_num_tx_ant,
                 channel_num_rx_ant,
                 num_bits_per_symbol,
                 ebno_db=None,
                 ebno_db_min=None,
                 ebno_db_max=None,
                 fec_num_iter=6
                 ):
        super().__init__()
        self.fec_type = fec_type
        self._scenario = scenario
        self._perfect_csi = perfect_csi

        # Internally set parameters
        self._carrier_frequency = 2.6e9 # 3.5e9
        self._fft_size = 36 # 128
        self._subcarrier_spacing = 15e3 # 30e3
        self._num_ofdm_symbols = 12 # 14
        self._cyclic_prefix_length = 6 # 20
        self._pilot_ofdm_symbol_indices = [2, 11]
        self._num_bs_ant = int(channel_num_rx_ant)
        self._num_ut = 1 # number of users communicating with a base station.
        self._num_ut_ant = int(channel_num_tx_ant) # number of antennas in a user terminal.
        self._num_bits_per_symbol = int(num_bits_per_symbol)
        self._coderate = 0.5

        self._dc_null = True
        self._num_guard_carriers = [5, 6]

        # Create an RX-TX association matrix
        # rx_tx_association[i,j]=1 means that receiver i gets at least one stream
        # from transmitter j. Depending on the transmission direction (uplink or downlink),
        # the role of UT and BS can change.
        bs_ut_association = np.zeros([1, self._num_ut])
        bs_ut_association[0, :] = 1
        self._rx_tx_association = bs_ut_association
        self._num_tx = self._num_ut
        self._num_streams_per_tx = self._num_ut_ant


        # Setup an OFDM Resource Grid
        self._rg = ResourceGrid(num_ofdm_symbols=self._num_ofdm_symbols,
                                fft_size=self._fft_size,
                                subcarrier_spacing=self._subcarrier_spacing,
                                num_tx=self._num_tx,
                                num_streams_per_tx=self._num_streams_per_tx,
                                cyclic_prefix_length=self._cyclic_prefix_length,
                                num_guard_carriers=self._num_guard_carriers,
                                dc_null=self._dc_null,
                                pilot_pattern="kronecker",
                                pilot_ofdm_symbol_indices=self._pilot_ofdm_symbol_indices)

        # Setup StreamManagement
        self._sm = StreamManagement(self._rx_tx_association, self._num_streams_per_tx)

        # Configure antenna arrays
        self._ut_array = AntennaArray(
                                 num_rows=1,
                                 num_cols=1,
                                 polarization="single",
                                 polarization_type="V",
                                 antenna_pattern="omni",
                                 carrier_frequency=self._carrier_frequency)

        self._bs_array = AntennaArray(num_rows=1,
                                      num_cols=int(self._num_bs_ant/2),
                                      polarization="dual",
                                      polarization_type="cross",
                                      antenna_pattern="38.901",
                                      carrier_frequency=self._carrier_frequency)

        # Configure the channel model
        if self._scenario == "umi":
            self._channel_model = UMi(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "uma":
            self._channel_model = UMa(carrier_frequency=self._carrier_frequency,
                                      o2i_model="low",
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)
        elif self._scenario == "rma":
            self._channel_model = RMa(carrier_frequency=self._carrier_frequency,
                                      ut_array=self._ut_array,
                                      bs_array=self._bs_array,
                                      direction="uplink",
                                      enable_pathloss=False,
                                      enable_shadow_fading=False)

        # Instantiate other building blocks
        self._qam_source = QAMSource(self._num_bits_per_symbol)

        # log
        print(f'{self._rg.num_data_symbols=}')
        # self._rg.num_data_symbols = self.num_effective_subcarriers * self._num_ofdm_symbols - \
        #       self.num_pilot_symbols
        print(f'{self._rg.num_effective_subcarriers=}')
        print(f'{self._rg._num_ofdm_symbols=}')
        print(f'{self._rg.num_pilot_symbols=}')
        # self._rg.num_effective_subcarriers= self._fft_size - self._dc_null - np.sum(self._num_guard_carriers)
        print(f'{self._rg._fft_size=}')
        print(f'{self._rg._dc_null=}')
        print(f'{np.sum(self._rg._num_guard_carriers)=}')

        self._n = int(self._rg.num_data_symbols*self._num_bits_per_symbol) # Number of coded bits
        self._k = int(self._n*self._coderate)                              # Number of information bits

        # log
        print(f'{self._k=}')
        print(f'{self._n=}')
        print(f'{self._coderate=}')

        print(f'{fec_type=}')
        print(f'{scenario=}')
        print(f'{perfect_csi=}')

        # FEC
        if self.fec_type == 'Polar5G':
            self._encoder = Polar5GEncoder(self._k, self._n)
            self._decoder = Polar5GDecoder(
                self._encoder,
                dec_type='SC',
                list_size=8
            )
        elif self.fec_type == 'LDPC5G':
            self.fec_num_iter = fec_num_iter
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True, num_iter=self.fec_num_iter)
        else:
            raise ValueError(f"Invalid channel coding type: {fec_type}")
        self._mapper = Mapper("qam", self._num_bits_per_symbol)
        self._rg_mapper = ResourceGridMapper(self._rg)

        self._ofdm_channel = OFDMChannel(self._channel_model, self._rg, add_awgn=True,
                                         normalize_channel=True, return_channel=True)

        self._remove_nulled_subcarriers = RemoveNulledSubcarriers(self._rg)
        self._ls_est = LSChannelEstimator(self._rg, interpolation_type="nn")
        self._lmmse_equ = LMMSEEqualizer(self._rg, self._sm)
        self._demapper = Demapper("app", "qam", self._num_bits_per_symbol)

        # channel noise
        assert ebno_db is not None or (ebno_db_min is not None and ebno_db_max is not None), "Set a single ebno_db or (ebno_db_min and ebno_db_max)"
        if ebno_db is not None:
            self.ebno_db = float(ebno_db)
        else:
            self.ebno_db = ebno_db # None
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max

        print(f'{self.ebno_db=}')
        print(f'{self.ebno_db_min=}')
        print(f'{self.ebno_db_max=}')
    
    def new_topology(self, batch_size):
        """Set new topology"""
        topology = gen_topology(batch_size,
                                self._num_ut,
                                self._scenario,
                                min_ut_velocity=0.0,
                                max_ut_velocity=0.0)

        self._channel_model.set_topology(*topology)
        

    @tf.function(jit_compile=True)
    def call(self, input):
        """
        Input
        -----
            :param input: 
        
        Output
        ------
            :return b_hat: 
        """
        # reshape input
        input_shape = input.shape

        divisor=self._num_tx * self._num_streams_per_tx * self._k
        if np.prod(input_shape) % divisor != 0:
            flatten_input = tf.reshape(input, [-1])
            flatten_input_len = len(flatten_input)
            
            dummy_cnt = ((flatten_input_len // divisor)+1) * divisor - flatten_input_len
            flatten_input = tf.concat([flatten_input, [0 for _ in range(dummy_cnt)]],0)
        else:
            flatten_input = input

        b = tf.reshape(flatten_input, (-1, self._num_tx, self._num_streams_per_tx, self._k))
        batch_size = b.shape[0]

        self.new_topology(batch_size)
        if self.ebno_db_min is not None and self.ebno_db_max is not None:
            ebno_db_tf = tf.random.uniform(shape=[batch_size], minval=self.ebno_db_min, maxval=self.ebno_db_max)
            no = ebnodb2no(ebno_db_tf, self._num_bits_per_symbol, self._coderate, self._rg)
        else:
            no = ebnodb2no(self.ebno_db, self._num_bits_per_symbol, self._coderate, self._rg)
        
        c = self._encoder(b)
        x = self._mapper(c)
        x_rg = self._rg_mapper(x)
        y, h = self._ofdm_channel([x_rg, no])
        if self._perfect_csi:
            h_hat = self._remove_nulled_subcarriers(h)
            err_var = 0.0
        else:
            h_hat, err_var = self._ls_est ([y, no])
        x_hat, no_eff = self._lmmse_equ([y, h_hat, err_var, no])
        llr = self._demapper([x_hat, no_eff])
        b_hat = self._decoder(llr)

        if np.prod(input_shape) % divisor != 0:
            #Reshape b_hat to the original shape by cutting the arbitrarily appended elements
            flatten_b_hat = tf.reshape(b_hat, [-1])
            sliced_b_hat = flatten_b_hat[:-dummy_cnt]
            b_hat=tf.reshape(sliced_b_hat, input_shape)
        else:
            b_hat=tf.reshape(b_hat, input_shape)
        
        return b_hat
        
class FlatFadingModel(tf.keras.Model):
    """
    Configure FlatFading Channel(for a simulation over RayLeigh) components.
    ref: https://nvlabs.github.io/sionna/examples/Simple_MIMO_Simulation.html?highlight=rayleigh 

    Parameters
    ----------
        :param fec_type: str, One of ["Polar5G", "LDPC5G"]
        :param channel_num_tx_ant: int
        :param channel_num_rx_ant: int
        :param num_bits_per_symbol: int
        :param fec_n: int
        :param fec_k: int 
        :param ebno_db: float
        :param ebno_db_min: float
        :param ebno_db_max: float
        :param fec_num_iter: int

    """
    def __init__(self,
                 fec_type,
                 channel_num_tx_ant,
                 channel_num_rx_ant,
                 num_bits_per_symbol,
                 fec_n,
                 fec_k,
                 ebno_db=None,
                 ebno_db_min=None,
                 ebno_db_max=None,
                 fec_num_iter=6
                 ):
        super().__init__()
        self.fec_type = fec_type
        
        self._n = fec_n
        self._k = fec_k
        self._coderate = self._k / self._n
        
        print(f'{self._k=}')
        print(f'{self._n=}')
        print(f'{self._coderate=}')

        constellation = Constellation("qam",
                                    num_bits_per_symbol,
                                    trainable=False)
        logger.info(f'Constellation: type={constellation._constellation_type} ' + \
                    f'{num_bits_per_symbol=} trainable={constellation._trainable}')
        self.num_bits_per_symbol = num_bits_per_symbol
        self.mapper = Mapper(constellation=constellation)
        
        self.channel_num_tx_ant = int(channel_num_tx_ant)
        self.channel_num_rx_ant = int(channel_num_rx_ant)
        self.channel = FlatFadingChannel(self.channel_num_tx_ant, self.channel_num_rx_ant, add_awgn=True, return_channel=True)

        # channel noise
        assert ebno_db is not None or (ebno_db_min is not None and ebno_db_max is not None), "Set a single ebno_db or (ebno_db_min and ebno_db_max)"
        if ebno_db is not None:
            self.ebno_db = float(ebno_db)
        else:
            self.ebno_db = ebno_db # None
        self.ebno_db_min = ebno_db_min
        self.ebno_db_max = ebno_db_max

        print(f'{self.ebno_db=}')
        print(f'{self.ebno_db_min=}')
        print(f'{self.ebno_db_max=}')

        self.demapper = Demapper("app", constellation=constellation)
        
        # FEC
        if self.fec_type == 'Polar5G':
            self._encoder = Polar5GEncoder(self._k, self._n)
            self._decoder = Polar5GDecoder(
                self._encoder,
                dec_type='SC',
                list_size=8
            )
        elif self.fec_type == 'LDPC5G':
            self.fec_num_iter = fec_num_iter
            self._encoder = LDPC5GEncoder(self._k, self._n)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=True, num_iter=self.fec_num_iter)
        else:
            raise ValueError(f"Invalid channel coding type: {fec_type}")
    
        
    @tf.function(jit_compile=True)
    def call(self, input):
        '''
        Input
        -----
            :param input: 
        
        Output
        ------
            :return b_hat: 
        '''
        # reshape input
        input_shape = input.shape
        divisor=self._k
        if np.prod(input_shape) % divisor != 0:
            flatten_input = tf.reshape(input, [-1])
            flatten_input_len = len(flatten_input)
            
            dummy_cnt = ((flatten_input_len // divisor)+1) * divisor - flatten_input_len
            flatten_input = tf.concat([flatten_input, [0 for _ in range(dummy_cnt)]],0)
        else:
            flatten_input = input
        
        # Channel encoder        
        b = tf.reshape(flatten_input, (-1, self.channel_num_tx_ant, self._k))
        codewords = self._encoder(b)
        
        # Modulation
        x = self.mapper(codewords)
        shape = tf.shape(x)
        x = tf.reshape(x, (-1, self.channel_num_tx_ant))

        #####################
        # Channel
        #####################
        # Sampling a batch of SNRs
        batch_size=b.shape[0]
        if self.ebno_db_min is not None and self.ebno_db_max is not None:
            ebno_db_tf = tf.random.uniform(shape=[batch_size], minval=self.ebno_db_min, maxval=self.ebno_db_max)
            no = ebnodb2no(ebno_db_tf, self.num_bits_per_symbol, self._coderate)
        else:
            no = ebnodb2no(self.ebno_db, self.num_bits_per_symbol, self._coderate)

        no *= np.sqrt(self.channel_num_rx_ant)

        y, h = self.channel([x, no])
        s = tf.complex(no*tf.eye(self.channel_num_rx_ant, self.channel_num_rx_ant), 0.0)

        # x_hat, no_eff = mf_equalizer(y, h, s)
        x_hat, no_eff = lmmse_equalizer(y, h, s)

        x_hat = tf.reshape(x_hat, shape)
        no_eff = tf.reshape(no_eff, shape)

        #####################
        # Receiver
        #####################
        # Demodulation
        llr = self.demapper([x_hat, no_eff])
        # llr = tf.reshape(llr, (-1, self._n))

        # Channel decoder
        b_hat = self._decoder(llr)

        if np.prod(input_shape) % divisor != 0:
            #Reshape b_hat to the original shape by cutting the arbitrarily appended elements
            flatten_b_hat = tf.reshape(b_hat, [-1])
            sliced_b_hat = flatten_b_hat[:-dummy_cnt]
            b_hat=tf.reshape(sliced_b_hat, input_shape)
        else:
            b_hat=tf.reshape(b_hat, input_shape)

        return b_hat