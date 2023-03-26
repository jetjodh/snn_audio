import librosa
from matplotlib import pyplot as plt
import numpy as np
import scipy


def reshape_signal_batch(signal):
    """Convert the signal into a standard batch shape for use with cochleagram.py
  functions. The first dimension is the batch dimension.
  https://github.com/mcdermottLab/pycochleagram/blob/master/pycochleagram/cochleagram.py
  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      either a flattened array with shape (n_samples,), a row vector with shape
      (1, n_samples), a column vector with shape (n_samples, 1), or a 2D
      matrix of the form [batch, waveform].
  Returns:
    array:
    **out_signal**: If the input `signal` has a valid shape, returns a
      2D version of the signal with the first dimension as the batch
      dimension.
  Raises:
    ValueError: Raises an error of the input `signal` has invalid shape.
  """
    if signal.ndim == 1:  # signal is a flattened array
        out_signal = signal.reshape((1, -1))
    elif signal.ndim == 2:  # signal is a row or column vector
        if signal.shape[0] == 1:
            out_signal = signal
        elif signal.shape[1] == 1:
            out_signal = signal.reshape((1, -1))
        else:  # first dim is batch dim
            out_signal = signal
    else:
        raise ValueError(
            "signal should be flat array, row or column vector, or a 2D matrix with dimensions [batch, waveform]; found %s"
            % signal.ndim
        )
    return out_signal


def freq2erb(freq_hz):
    """Converts Hz to human-defined ERBs, using the formula of Glasberg
  and Moore.
  Args:
    freq_hz (array_like): frequency to use for ERB.
  Returns:
    ndarray:
    **n_erb** -- Human-defined ERB representation of input.
  """
    return 9.265 * np.log(1 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    """Converts human ERBs to Hz, using the formula of Glasberg and Moore.
  Args:
    n_erb (array_like): Human-defined ERB to convert to frequency.
  Returns:
    ndarray:
    **freq_hz** -- Frequency representation of input.
  """
    return 24.7 * 9.265 * (np.exp(n_erb / 9.265) - 1)


def make_cosine_filter(freqs, l, h, convert_to_erb=True):
    """Generate a half-cosine filter. Represents one subband of the cochleagram.
  A half-cosine filter is created using the values of freqs that are within the
  interval [l, h]. The half-cosine filter is centered at the center of this
  interval, i.e., (h - l) / 2. Values outside the valid interval [l, h] are
  discarded. So, if freqs = [1, 2, 3, ... 10], l = 4.5, h = 8, the cosine filter
  will only be defined on the domain [5, 6, 7] and the returned output will only
  contain 3 elements.
  Args:
    freqs (array_like): Array containing the domain of the filter, in ERB space;
      see convert_to_erb parameter below.. A single half-cosine
      filter will be defined only on the valid section of these values;
      specifically, the values between cutoffs ``l`` and ``h``. A half-cosine filter
      centered at (h - l ) / 2 is created on the interval [l, h].
    l (float): The lower cutoff of the half-cosine filter in ERB space; see
      convert_to_erb parameter below.
    h (float): The upper cutoff of the half-cosine filter in ERB space; see
      convert_to_erb parameter below.
    convert_to_erb (bool, default=True): If this is True, the values in
      input arguments ``freqs``, ``l``, and ``h`` will be transformed from Hz to ERB
      space before creating the half-cosine filter. If this is False, the
      input arguments are assumed to be in ERB space.
  Returns:
    ndarray:
    **half_cos_filter** -- A half-cosine filter defined using elements of
    freqs within [l, h].
  """
    if convert_to_erb:
        freqs_erb = freq2erb(freqs)
        l_erb = freq2erb(l)
        h_erb = freq2erb(h)
    else:
        freqs_erb = freqs
        l_erb = l
        h_erb = h

    avg_in_erb = (l_erb + h_erb) / 2  # center of filter
    rnge_in_erb = h_erb - l_erb  # width of filter
    return np.cos(
        (freqs_erb[(freqs_erb > l_erb) & (freqs_erb < h_erb)] - avg_in_erb)
        / rnge_in_erb
        * np.pi
    )  # map cutoffs to -pi/2, pi/2 interval


def make_erb_cos_filters_nx(
    signal_length,
    sr,
    n,
    low_lim,
    hi_lim,
    sample_factor,
    padding_size=None,
    full_filter=True,
    strict=True,
    **kwargs
):
    """Create ERB cosine filters, oversampled by a factor provided by "sample_factor"
  Args:
    signal_length (int): Length of signal to be filtered with the generated
      filterbank. The signal length determines the length of the filters.
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor). The
      number of filters in the generated filterbank is given below:
      +---------------+---------------+-+------------+---+---------------------+
      | sample factor |     n_out     |=|  bandpass  |\ +|  highpass + lowpass |
      +===============+===============+=+============+===+=====================+
      |      1        |     n+2       |=|     n      |\ +|     1     +    1    |
      +---------------+---------------+-+------------+---+---------------------+
      |      2        |   2*n+1+4     |=|   2*n+1    |\ +|     2     +    2    |
      +---------------+---------------+-+------------+---+---------------------+
      |      4        |   4*n+3+8     |=|   4*n+3    |\ +|     4     +    4    |
      +---------------+---------------+-+------------+---+---------------------+
      |      s        | s*(n+1)-1+2*s |=|  s*(n+1)-1 |\ +|     s     +    s    |
      +---------------+---------------+-+------------+---+---------------------+
    low_lim (int): Lower limit of frequency range. Filters will not be defined
      below this limit.
    hi_lim (int): Upper limit of frequency range. Filters will not be defined
      above this limit.
    sample_factor (int): Positive integer that determines how densely ERB function
     will be sampled to create bandpass filters. 1 represents standard sampling;
     adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
     adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
     adjacent bandpass filters will overlap by 87.5%.
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, default=True): If True (default), the complete filter that
      is ready to apply to the signal is returned. If False, only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, default=True): If True (default), will throw an error if
      sample_factor is not a power of two. This facilitates comparison across
      sample_factors. Also, if True, will throw an error if provided hi_lim
      is greater than the Nyquist rate.
  Returns:
    tuple:
    A tuple containing the output:
      * **filts** (*array*)-- The filterbank consisting of filters have
        cosine-shaped frequency responses, with center frequencies equally
        spaced on an ERB scale from low_lim to hi_lim.
      * **center_freqs** (*array*) -- something
      * **freqs** (*array*) -- something
  Raises:
    ValueError: Various value errors for bad choices of sample_factor; see
      description for strict parameter.
  """
    if not isinstance(sample_factor, int):
        raise ValueError(
            "sample_factor must be an integer, not %s" % type(sample_factor)
        )
    if sample_factor <= 0:
        raise ValueError("sample_factor must be positive")

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = "sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison."

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    # verify the high limit is allowed by the sampling rate
    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'

    # changing the sampling density without changing the filter locations
    # (and, thereby changing their widths) requires that a certain number of filters
    # be used.
    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = np.linspace(0, max_freq, n_freqs)

    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))  # ?? n_freqs+1

    # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
    # get the actual spacing use to generate the sequence (in case numpy does something weird)
    center_freqs, erb_spacing = np.linspace(
        freq2erb(low_lim), freq2erb(hi_lim), n_filters + 2, retstep=True
    )  # +2 for bin endpoints
    # we need to exclude the endpoints
    center_freqs = center_freqs[1:-1]

    freqs_erb = freq2erb(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * erb_spacing
        h = center_freqs[i] + sample_factor * erb_spacing
        # the first sample_factor # of rows in filts will be lowpass filters
        filts[(freqs_erb > l) & (freqs_erb < h), i_offset] = make_cosine_filter(
            freqs_erb, l, h, convert_to_erb=False
        )

    # be sample_factor number of each
    for i in range(sample_factor):
        # account for the fact that the first sample_factor # of filts are lowpass
        i_offset = i + sample_factor
        lp_h_ind = max(
            np.where(freqs < erb2freq(center_freqs[i]))[0]
        )  # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[: lp_h_ind + 1, i_offset], 2))

        hp_l_ind = min(
            np.where(freqs > erb2freq(center_freqs[-1 - i]))[0]
        )  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1 - i_offset], 2))

        filts[: lp_h_ind + 1, i] = lp_filt
        filts[hp_l_ind:, -1 - i] = hp_filt

    # ensure that squared freq response adds to one
    filts = filts / np.sqrt(sample_factor)

    # get center freqs for lowpass and highpass filters
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
    center_freqs = erb2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

    # rectify
    center_freqs[center_freqs < 0] = 1

    # discard highpass and lowpass filters, if requested
    if kwargs.get("no_lowpass"):
        filts = filts[:, sample_factor:]
    if kwargs.get("no_highpass"):
        filts = filts[:, :-sample_factor]

    # make the full filter by adding negative components
    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def make_full_filter_set(filts, signal_length=None):
    """Create the full set of filters by extending the filterbank to negative FFT
  frequencies.
  Args:
    filts (array_like): Array containing the cochlear filterbank in frequency space,
      i.e., the output of make_erb_cos_filters_nx. Each row of ``filts`` is a
      single filter, with columns indexing frequency.
    signal_length (int, optional): Length of the signal to be filtered with this filterbank.
      This should be equal to filter length * 2 - 1, i.e., 2*filts.shape[1] - 1, and if
      signal_length is None, this value will be computed with the above formula.
      This parameter might be deprecated later.
  Returns:
    ndarray:
    **full_filter_set** -- Array containing the complete filterbank in
    frequency space. This output can be directly applied to the frequency
    representation of a signal.
  """
    if signal_length is None:
        signal_length = 2 * filts.shape[1] - 1

    # note that filters are currently such that each ROW is a filter and COLUMN idxs freq
    if (
        np.remainder(signal_length, 2) == 0
    ):  # even -- don't take the DC & don't double sample nyquist
        neg_filts = np.flipud(filts[1 : filts.shape[0] - 1, :])
    else:  # odd -- don't take the DC
        neg_filts = np.flipud(filts[1 : filts.shape[0], :])
    fft_filts = np.vstack((filts, neg_filts))
    # we need to switch representation to apply filters to fft of the signal, not sure why, but do it here
    return fft_filts.T


def make_erb_cos_filters_nx(
    signal_length,
    sr,
    n,
    low_lim,
    hi_lim,
    sample_factor,
    padding_size=None,
    full_filter=True,
    strict=True,
    **kwargs
):
    """Create ERB cosine filters, oversampled by a factor provided by "sample_factor"
  Args:
    signal_length (int): Length of signal to be filtered with the generated
      filterbank. The signal length determines the length of the filters.
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor). The
      number of filters in the generated filterbank is given below:
      +---------------+---------------+-+------------+---+---------------------+
      | sample factor |     n_out     |=|  bandpass  |\ +|  highpass + lowpass |
      +===============+===============+=+============+===+=====================+
      |      1        |     n+2       |=|     n      |\ +|     1     +    1    |
      +---------------+---------------+-+------------+---+---------------------+
      |      2        |   2*n+1+4     |=|   2*n+1    |\ +|     2     +    2    |
      +---------------+---------------+-+------------+---+---------------------+
      |      4        |   4*n+3+8     |=|   4*n+3    |\ +|     4     +    4    |
      +---------------+---------------+-+------------+---+---------------------+
      |      s        | s*(n+1)-1+2*s |=|  s*(n+1)-1 |\ +|     s     +    s    |
      +---------------+---------------+-+------------+---+---------------------+
    low_lim (int): Lower limit of frequency range. Filters will not be defined
      below this limit.
    hi_lim (int): Upper limit of frequency range. Filters will not be defined
      above this limit.
    sample_factor (int): Positive integer that determines how densely ERB function
     will be sampled to create bandpass filters. 1 represents standard sampling;
     adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
     adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
     adjacent bandpass filters will overlap by 87.5%.
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size*signal_length.
    full_filter (bool, default=True): If True (default), the complete filter that
      is ready to apply to the signal is returned. If False, only the first
      half of the filter is returned (likely positive terms of FFT).
    strict (bool, default=True): If True (default), will throw an error if
      sample_factor is not a power of two. This facilitates comparison across
      sample_factors. Also, if True, will throw an error if provided hi_lim
      is greater than the Nyquist rate.
  Returns:
    tuple:
    A tuple containing the output:
      * **filts** (*array*)-- The filterbank consisting of filters have
        cosine-shaped frequency responses, with center frequencies equally
        spaced on an ERB scale from low_lim to hi_lim.
      * **center_freqs** (*array*) -- something
      * **freqs** (*array*) -- something
  Raises:
    ValueError: Various value errors for bad choices of sample_factor; see
      description for strict parameter.
  """
    if not isinstance(sample_factor, int):
        raise ValueError(
            "sample_factor must be an integer, not %s" % type(sample_factor)
        )
    if sample_factor <= 0:
        raise ValueError("sample_factor must be positive")

    if sample_factor != 1 and np.remainder(sample_factor, 2) != 0:
        msg = "sample_factor odd, and will change ERB filter widths. Use even sample factors for comparison."

    if padding_size is not None and padding_size >= 1:
        signal_length += padding_size

    if np.remainder(signal_length, 2) == 0:  # even length
        n_freqs = signal_length // 2  # .0 does not include DC, likely the sampling grid
        max_freq = sr / 2  # go all the way to nyquist
    else:  # odd length
        n_freqs = (signal_length - 1) // 2  # .0
        max_freq = sr * (signal_length - 1) / 2 / signal_length  # just under nyquist

    # verify the high limit is allowed by the sampling rate
    if hi_lim > sr / 2:
        hi_lim = max_freq
        msg = 'input arg "hi_lim" exceeds nyquist limit for max frequency; ignore with "strict=False"'

    # changing the sampling density without changing the filter locations
    # (and, thereby changing their widths) requires that a certain number of filters
    # be used.
    n_filters = sample_factor * (n + 1) - 1
    n_lp_hp = 2 * sample_factor
    freqs = np.linspace(0, max_freq, n_freqs)
    filts = np.zeros((n_freqs + 1, n_filters + n_lp_hp))  # ?? n_freqs+1

    # cutoffs are evenly spaced on an erb scale -- interpolate linearly in erb space then convert back
    # get the actual spacing use to generate the sequence (in case numpy does something weird)
    center_freqs, erb_spacing = np.linspace(
        freq2erb(low_lim), freq2erb(hi_lim), n_filters + 2, retstep=True
    )  # +2 for bin endpoints
    # we need to exclude the endpoints
    center_freqs = center_freqs[1:-1]

    freqs_erb = freq2erb(freqs)
    for i in range(n_filters):
        i_offset = i + sample_factor
        l = center_freqs[i] - sample_factor * erb_spacing
        h = center_freqs[i] + sample_factor * erb_spacing
        # the first sample_factor # of rows in filts will be lowpass filters
        filts[:-1][(freqs_erb > l) & (freqs_erb < h), i_offset] = make_cosine_filter(
            freqs_erb, l, h, convert_to_erb=False
        )

    # be sample_factor number of each
    for i in range(sample_factor):
        # account for the fact that the first sample_factor # of filts are lowpass
        i_offset = i + sample_factor
        lp_h_ind = max(
            np.where(freqs < erb2freq(center_freqs[i]))[0]
        )  # lowpass filter goes up to peak of first cos filter
        lp_filt = np.sqrt(1 - np.power(filts[: lp_h_ind + 1, i_offset], 2))

        hp_l_ind = min(
            np.where(freqs > erb2freq(center_freqs[-1 - i]))[0]
        )  # highpass filter goes down to peak of last cos filter
        hp_filt = np.sqrt(1 - np.power(filts[hp_l_ind:, -1 - i_offset], 2))

        filts[: lp_h_ind + 1, i] = lp_filt
        filts[hp_l_ind:, -1 - i] = hp_filt

    # ensure that squared freq response adds to one
    filts = filts / np.sqrt(sample_factor)

    # get center freqs for lowpass and highpass filters
    cfs_low = np.copy(center_freqs[:sample_factor]) - sample_factor * erb_spacing
    cfs_hi = np.copy(center_freqs[-sample_factor:]) + sample_factor * erb_spacing
    center_freqs = erb2freq(np.concatenate((cfs_low, center_freqs, cfs_hi)))

    # rectify
    center_freqs[center_freqs < 0] = 1

    # discard highpass and lowpass filters, if requested
    if kwargs.get("no_lowpass"):
        filts = filts[:, sample_factor:]
    if kwargs.get("no_highpass"):
        filts = filts[:, :-sample_factor]

    # make the full filter by adding negative components
    if full_filter:
        filts = make_full_filter_set(filts, signal_length)

    return filts, center_freqs, freqs


def apply_envelope_downsample(
    subband_envelopes, mode, audio_sr=None, env_sr=None, invert=False, strict=True
):
    """Apply a downsampling operation to cochleagram subband envelopes.
  The `mode` argument can be a predefined downsampling type from
  {'poly', 'resample', 'decimate'}, a callable (to perform custom downsampling),
  or None to return the unmodified cochleagram. If `mode` is a predefined type,
  `audio_sr` and `env_sr` are required.
  Args:
    subband_envelopes (array): Cochleagram subbands to mode.
    mode ({'poly', 'resample', 'decimate', callable, None}): Determines the
      downsampling operation to apply to the cochleagram. 'decimate' will
      resample using scipy.signal.decimate with audio_sr/env_sr as the
      downsampling factor. 'resample' will downsample using
      scipy.signal.resample with np.ceil(subband_envelopes.shape[1]*(audio_sr/env_sr))
      as the number of samples. 'poly' will resample using scipy.signal.resample_poly
      with `env_sr` as the upsampling factor and `audio_sr` as the downsampling
      factor. If `mode` is a python callable (e.g., function), it will be
      applied to `subband_envelopes`. If this is None, no  downsampling is
      performed and the unmodified cochleagram is returned.
    audio_sr (int, optional): If using a predefined sampling `mode`, this
      represents the sampling rate of the original signal.
    env_sr (int, optional): If using a predefined sampling `mode`, this
      represents the sampling rate of the downsampled subband envelopes.
    invert (bool, optional):  If using a predefined sampling `mode`, this
      will invert (i.e., upsample) the subband envelopes using the values
      provided in `audio_sr` and `env_sr`.
    strict (bool, optional): If using a predefined sampling `mode`, this
      ensure the downsampling will result in an integer number of samples. This
      should mean the upsample(downsample(x)) will have the same number of
      samples as x.
  Returns:
    array:
    **downsampled_subband_envelopes**: The subband_envelopes after being
      downsampled with `mode`.
  """
    if mode is None:
        pass
    elif callable(mode):
        # apply the downsampling function
        subband_envelopes = mode(subband_envelopes)
    else:
        mode = mode.lower()
        if audio_sr is None:
            raise ValueError(
                "`audio_sr` cannot be None. Provide sampling rate of original audio signal."
            )
        if env_sr is None:
            raise ValueError(
                "`env_sr` cannot be None. Provide sampling rate of subband envelopes (cochleagram)."
            )

        if mode == "decimate":
            if invert:
                raise NotImplementedError()
            else:
                # was BadCoefficients error with Chebyshev type I filter [default]
                subband_envelopes = scipy.signal.decimate(
                    subband_envelopes, audio_sr // env_sr, axis=1, ftype="fir"
                )  # this caused weird banding artifacts
        elif mode == "resample":
            if invert:
                subband_envelopes = scipy.signal.resample(
                    subband_envelopes,
                    np.ceil(subband_envelopes.shape[1] * (audio_sr / env_sr)),
                    axis=1,
                )  # fourier method: this causes NANs that get converted to 0s
            else:
                subband_envelopes = scipy.signal.resample(
                    subband_envelopes,
                    np.ceil(subband_envelopes.shape[1] * (env_sr / audio_sr)),
                    axis=1,
                )  # fourier method: this causes NANs that get converted to 0s
        elif mode == "poly":
            if strict:
                n_samples = (
                    subband_envelopes.shape[1] * (audio_sr / env_sr)
                    if invert
                    else subband_envelopes.shape[1] * (env_sr / audio_sr)
                )
                if not np.isclose(n_samples, int(n_samples)):
                    raise ValueError(
                        "Choose `env_sr` and `audio_sr` such that the number of samples after polyphase resampling is an integer"
                        + "\n(length: %s, env_sr: %s, audio_sr: %s !--> %s"
                        % (subband_envelopes.shape[1], env_sr, audio_sr, n_samples)
                    )
            if invert:
                subband_envelopes = scipy.signal.resample_poly(
                    subband_envelopes, audio_sr, env_sr, axis=1
                )  # this requires v0.18 of scipy
            else:
                subband_envelopes = scipy.signal.resample_poly(
                    subband_envelopes, env_sr, audio_sr, axis=1
                )  # this requires v0.18 of scipy
        else:
            raise ValueError("Unsupported downsampling `mode`: %s" % mode)
    subband_envelopes[subband_envelopes < 0] = 0
    return subband_envelopes


def apply_envelope_nonlinearity(subband_envelopes, nonlinearity, invert=False):
    """Apply a nonlinearity to the cochleagram.
  The `nonlinearity` argument can be an predefined type, a callable
  (to apply a custom nonlinearity), or None to return the unmodified
  cochleagram.
  Args:
    subband_envelopes (array): Cochleagram to apply the nonlinearity to.
    nonlinearity ({'db', 'power'}, callable, None): Determines the nonlinearity
      operation to apply to the cochleagram. If this is a valid string, one
      of the predefined nonlinearities will be used. It can be: 'power' to
      perform np.power(subband_envelopes, 3.0 / 10.0) or 'db' to perform
      20 * np.log10(subband_envelopes / np.max(subband_envelopes)), with values
      clamped to be greater than -60. If `nonlinearity` is a python callable
      (e.g., function), it will be applied to `subband_envelopes`. If this is
      None, no nonlinearity is applied and the unmodified cochleagram is
      returned.
    invert (bool): For predefined nonlinearities 'db' and 'power', if False
      (default), the nonlinearity will be applied. If True, the nonlinearity
      will be inverted.
  Returns:
    array:
    **nonlinear_subband_envelopes**: The subband_envelopes with the specified
      nonlinearity applied.
  Raises:
      ValueError: Error if the provided `nonlinearity` isn't a recognized
      option.
  """
    # apply nonlinearity
    if nonlinearity is None:
        pass
    elif nonlinearity == "power":
        if invert:
            subband_envelopes = np.power(
                subband_envelopes, 10.0 / 3.0
            )  # from Alex's code
        else:
            subband_envelopes = np.power(
                subband_envelopes, 3.0 / 10.0
            )  # from Alex's code
    elif nonlinearity == "db":
        if invert:
            subband_envelopes = np.power(
                10, subband_envelopes / 20
            )  # adapted from Anastasiya's code
        else:
            dtype_eps = np.finfo(subband_envelopes.dtype).eps
            subband_envelopes[subband_envelopes == 0] = dtype_eps
            subband_envelopes = 20 * np.log10(
                subband_envelopes / np.max(subband_envelopes)
            )
            subband_envelopes[subband_envelopes < -60] = -60
    elif callable(nonlinearity):
        subband_envelopes = nonlinearity(subband_envelopes)
    else:
        raise ValueError(
            'argument "nonlinearity" must be "power", "db", or a function.'
        )
    return subband_envelopes


def reshape_signal_canonical(signal):
    """Convert the signal into a canonical shape for use with cochleagram.py
  functions.
  This first verifies that the signal contains only one data channel, which can
  be in a row, a column, or a flat array. Then it flattens the signal array.
  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      either a flattened array with shape (n_samples,), a row vector with shape
      (1, n_samples), or a column vector with shape (n_samples, 1).
  Returns:
    array:
    **out_signal**: If the input `signal` has a valid shape, returns a
      flattened version of the signal.
  Raises:
    ValueError: Raises an error of the input `signal` has invalid shape.
  """
    if signal.ndim == 1:  # signal is a flattened array
        out_signal = signal
    elif signal.ndim == 2:  # signal is a row or column vector
        if signal.shape[0] == 1:
            out_signal = signal.flatten()
        elif signal.shape[1] == 1:
            out_signal = signal.flatten()
        else:
            raise ValueError(
                "signal must be a row or column vector; found shape: %s" % signal.shape
            )
    else:
        raise ValueError(
            "signal must be a row or column vector; found shape: %s" % signal.shape
        )
    return out_signal


def pad_signal(signal, padding_size, axis=0):
    """Pad the signal by appending zeros to the end. The padded signal has
  length `padding_size * length(signal)`.
  Args:
    signal (array): The signal to be zero-padded.
    padding_size (int): Factor that determines the size of the padded signal.
      The padded signal has length `padding_size * length(signal)`.
    axis (int): Specifies the axis to pad; defaults to 0.
  Returns:
    tuple:
      **pad_signal** (*array*): The zero-padded signal.
      **padding_size** (*int*): The length of the zero-padding added to the array.
  """
    if padding_size is not None and padding_size >= 1:
        pad_shape = list(signal.shape)
        pad_shape[axis] = padding_size
        pad_signal = np.concatenate((signal, np.zeros(pad_shape)))
    else:
        padding_size = 0
        pad_signal = signal
    return (pad_signal, padding_size)


def rfft(a, n=None, axis=-1, mode="auto", params=None):
    """Provides support for various implementations of the RFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.
  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n` and `axis`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    array:
    **rfft_a**: Signal in the frequency domain in standard order.
      See numpy.rfft() for a description of the output.
  """
    # handle 'auto' mode
    mode, params = _parse_fft_mode(mode, params)
    # named args override params
    d1 = {"n": n, "axis": axis}
    params = dict(d1, **params)

    if mode == "fftw":
        import pyfftw

        return pyfftw.interfaces.numpy_fft.rfft(a, **params)
    elif mode == "np":
        return np.fft.rfft(a, **params)
    else:
        raise NotImplementedError(
            "`rfft method is not defined for mode `%s`;" + 'use "np" or "fftw".'
        )


def fft(a, n=None, axis=-1, norm=None, mode="auto", params=None):
    """Provides support for various implementations of the FFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.
  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    norm ({None, 'ortho'}, optional): Support for numpy interface.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n`, `axis`, and `norm`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    array:
      **fft_a**: Signal in the frequency domain in FFT standard order. See numpy.fft() for
      a description of the output.
  """
    # handle 'auto' mode
    mode, params = _parse_fft_mode(mode, params)
    # named args override params
    d1 = {"n": n, "axis": axis, "norm": norm}
    params = dict(d1, **params)

    if mode == "fftw":
        import pyfftw

        return pyfftw.interfaces.numpy_fft.fft(a, **params)
    elif mode == "np":
        return np.fft.fft(a, **params)
    else:
        raise NotImplementedError(
            "`fft method is not defined for mode `%s`;" + 'use "auto", "np" or "fftw".'
        )


def ifft(a, n=None, axis=-1, norm=None, mode="auto", params=None):
    """Provides support for various implementations of the IFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.
  Args:
    a (array): Time-domain signal.
    mode (str): Determines which IFFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    norm ({None, 'ortho'}, optional): Support for numpy interface.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n`, `axis`, and `norm`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    array:
    **ifft_a**: Signal in the time domain. See numpy.ifft() for a
      description of the output.
  """
    # handle 'auto' mode
    mode, params = _parse_fft_mode(mode, params)
    # named args override params
    d1 = {"n": n, "axis": axis, "norm": norm}
    params = dict(d1, **params)

    if mode == "fftw":
        import pyfftw

        return pyfftw.interfaces.numpy_fft.ifft(a, **params)
    elif mode == "np":
        return np.fft.ifft(a, **params)
    else:
        raise NotImplementedError(
            "`ifft method is not defined for mode `%s`;" + 'use "np" or "fftw".'
        )


def fhilbert(a, axis=None, mode="auto", ifft_params=None):
    """Compute the Hilbert transform of the provided frequency-space signal.
  This function assumes the input array is already in frequency space, i.e.,
  it is the output of a numpy-like FFT implementation. This avoids unnecessary
  repeated computation of the FFT/IFFT.
  Args:
    a (array): Signal, in frequency space, e.g., a = fft(signal).
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    iff_params (dict, None, optional): Dictionary of input arguments to provide to
      the call computing ifft. If `mode` is 'auto' and params dict is None,
      sensible values will be chosen. If `ifft_params` is not None, it will not
      be altered.
  Returns:
    array:
    **hilbert_a**: Hilbert transform of input array `a`, in the time domain.
  """
    if axis is None:
        axis = np.argmax(a.shape)
    N = a.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    # perform the hilbert transform in the frequency domain
    # algorithm from scipy.signal.hilbert
    h = np.zeros(N)  # don't modify the input array
    # create hilbert multiplier
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2
    ah = a * h  # apply hilbert transform

    return ifft(ah, mode=mode, params=ifft_params)


def generate_subband_envelopes_fast(
    signal, filters, padding_size=None, fft_mode="auto", debug_ret_all=False
):
    """Generate the subband envelopes (i.e., the cochleagram) of the signal by
  applying the provided filters.
  This method returns *only* the envelopes of the subband decomposition.
  The signal can be optionally zero-padded before the decomposition. The
  resulting envelopes can be optionally downsampled and then modified with a
  nonlinearity.
  This function expedites the calculation of the subbands envelopes by:
    1) using the rfft rather than standard fft to compute the dft for
       real-valued signals
    2) hand-computing the Hilbert transform, to avoid unnecessary calls
       to fft/ifft.
  See utils.rfft, utils.irfft, and utils.fhilbert for more details on the
  methods used for speed-up.
  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    padding_size (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `padding_size * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
  Returns:
    array:
    **subband_envelopes**: The subband envelopes (i.e., cochleagram) resulting from
      the subband decomposition. This should have the same shape as `filters`.
  """
    # convert the signal to a canonical representation
    signal_flat = reshape_signal_canonical(signal)

    if padding_size is not None and padding_size > 1:
        signal_flat, padding = pad_signal(signal_flat, padding_size)

    if np.isrealobj(signal_flat):  # attempt to speed up computation with rfft
        fft_sample = rfft(signal_flat, mode=fft_mode)
        nr = fft_sample.shape[0]
        # prep for hilbert transform by extending to negative freqs
        subbands = np.zeros(filters.shape, dtype=complex)
        subbands[:, :nr] = _real_freq_filter(fft_sample, filters)
    else:
        fft_sample = fft(signal_flat, mode=fft_mode)
        subbands = filters * fft_sample

    analytic_subbands = fhilbert(subbands, mode=fft_mode)
    subband_envelopes = np.abs(analytic_subbands)

    if padding_size is not None and padding_size > 1:
        analytic_subbands = analytic_subbands[
            :, : signal_flat.shape[0] - padding
        ]  # i dont know if this is correct
        subband_envelopes = subband_envelopes[
            :, : signal_flat.shape[0] - padding
        ]  # i dont know if this is correct

    if debug_ret_all is True:
        out_dict = {}
        # add all local variables to out_dict
        for k in dir():
            if k != "out_dict":
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return subband_envelopes


def fft(a, n=None, axis=-1, norm=None, mode="auto", params=None):
    """Provides support for various implementations of the FFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.
  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    norm ({None, 'ortho'}, optional): Support for numpy interface.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n`, `axis`, and `norm`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    array:
      **fft_a**: Signal in the frequency domain in FFT standard order. See numpy.fft() for
      a description of the output.
  """
    # handle 'auto' mode
    mode, params = _parse_fft_mode(mode, params)
    # named args override params
    d1 = {"n": n, "axis": axis, "norm": norm}
    params = dict(d1, **params)

    if mode == "np":
        return np.fft.fft(a, **params)
    else:
        raise NotImplementedError(
            "`fft method is not defined for mode `%s`;" + 'use "auto", "np" or "fftw".'
        )


def _parse_fft_mode(mode, params):
    """Prepare mode and params arguments provided by user for use with
  utils.fft, utils.ifft, etc.
  Args:
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    params (dict, None): Dictionary of input arguments to provide to the
      appropriate fft function. If `mode` is 'auto' and params dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    tuple:
      **out_mode** (str): The mode determining the fft implementation to use; either
        'np' or 'fftw'.
      **out_params** (dict): A dictionary containing input arguments to the
        fft function.
  """
    mode == mode.lower()
    if mode == "auto":
        try:
            import pyfftw

            mode = "fftw"
            if params is None:
                params = {"planner_effort": "FFTW_ESTIMATE"}  # FFTW_ESTIMATE seems fast
        except ImportError:
            mode = "np"
            if params is None:
                params = {}
    else:
        if params is None:
            params = {}
    return mode, params


def _real_freq_filter(rfft_signal, filters):
    """Helper function to apply a full filterbank to a rfft signal
  """
    nr = rfft_signal.shape[0]
    subbands = filters[:, :nr] * rfft_signal
    return subbands


def irfft(a, n=None, axis=-1, mode="auto", params=None):
    """Provides support for various implementations of the IRFFT, using numpy's
  fftpack or pyfftw's fftw. This uses a numpy.fft-like interface.
  Args:
    a (array): Time-domain signal.
    mode (str): Determines which FFT implementation will be used. Options are
      'fftw', 'np', and 'auto'. Using 'auto', will attempt to use a pyfftw
      implementation with some sensible parameters (if the module is
      available), and will use numpy's fftpack implementation otherwise.
    n (int, optional): Length of the transformed axis of the output. If n is
      smaller than the length of the input, the input is cropped. If it is
      larger, the input is padded with zeros. If n is not given, the length of
      the input along the axis specified by axis is used.
    axis (int, optional): Axis over which to compute the FFT. If not given, the
      last axis is used.
    params (dict, None, optional): Dictionary of additional input arguments to
      provide to the appropriate fft function (usually fftw). Note, named
      arguments (e.g., `n` and `axis`) will override identically named
      arguments in `params`. If `mode` is 'auto' and `params` dict is None,
      sensible values will be chosen. If `params` is not None, it will not be
      altered.
  Returns:
    array:
    **irfft_a**: Signal in the time domain. See numpy.irfft() for a
      description of the output.
  """
    # handle 'auto' mode
    mode, params = _parse_fft_mode(mode, params)
    # named args override params
    # d1 = {'n': n, 'axis': axis, 'norm': norm}
    d1 = {"n": n, "axis": axis}
    params = dict(d1, **params)

    if mode == "fftw":
        import pyfftw

        return pyfftw.interfaces.numpy_fft.irfft(a, **params)
    elif mode == "np":
        return np.fft.irfft(a, **params)
    else:
        raise NotImplementedError(
            "`irfft method is not defined for mode `%s`;" + 'use "np" or "fftw".'
        )


def generate_subbands(
    signal, filters, padding_size=None, fft_mode="auto", debug_ret_all=False
):
    """Generate the subband decomposition of the signal by applying the provided
  filters.
  The input filters are applied to the signal to perform subband decomposition.
  The signal can be optionally zero-padded before the decomposition.
  Args:
    signal (array): The sound signal (waveform) in the time domain.
    filters (array): The filterbank, in frequency space, used to generate the
      cochleagram. This should be the full filter-set output of
      erbFilter.make_erb_cos_filters_nx, or similar.
    padding_size (int, optional): Factor that determines if the signal will be
      zero-padded before generating the subbands. If this is None,
      or less than 1, no zero-padding will be used. Otherwise, zeros are added
      to the end of the input signal until is it of length
      `padding_size * length(signal)`. This padded region will be removed after
      performing the subband decomposition.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
  Returns:
    array:
    **subbands**: The subbands resulting from the subband decomposition. This
      should have the same shape as `filters`.
  """

    # convert the signal to a canonical representation
    signal_flat = reshape_signal_canonical(signal)

    if padding_size is not None and padding_size > 1:
        signal_flat, padding = pad_signal(signal_flat, padding_size)

    is_signal_even = signal_flat.shape[0] % 2 == 0
    if (
        np.isrealobj(signal_flat) and is_signal_even
    ):  # attempt to speed up computation with rfft
        if signal_flat.shape[0] % 2 == 0:
            fft_sample = rfft(signal_flat, mode=fft_mode)
            subbands = _real_freq_filter(fft_sample, filters)
            subbands = irfft(subbands, mode=fft_mode)  # operates row-wise
        else:
            print(
                "Consider using even-length signal for a rfft speedup", RuntimeWarning
            )
            fft_sample = fft(signal_flat, mode=fft_mode)
            subbands = filters * fft_sample
            subbands = np.real(ifft(subbands, mode=fft_mode))  # operates row-wise
    else:
        fft_sample = fft(signal_flat, mode=fft_mode)
        subbands = filters * fft_sample
        subbands = np.real(ifft(subbands, mode=fft_mode))  # operates row-wise

    if padding_size is not None and padding_size > 1:
        subbands = subbands[
            :, : signal_flat.shape[0] - padding
        ]  # i dont know if this is correct

    if debug_ret_all is True:
        out_dict = {}
        # add all local variables to out_dict
        for k in dir():
            if k != "out_dict":
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return subbands


def cochleagram(
    signal,
    sr,
    n,
    low_lim,
    hi_lim,
    sample_factor,
    padding_size=None,
    downsample=None,
    nonlinearity=None,
    fft_mode="auto",
    ret_mode="envs",
    strict=True,
    **kwargs
):
    """Generate the subband envelopes (i.e., the cochleagram)
  of the provided signal.
  This first creates a an ERB filterbank with the provided input arguments for
  the provided signal. This filterbank is then used to perform the subband
  decomposition to create the subband envelopes. The resulting envelopes can be
  optionally downsampled and then modified with a nonlinearity.
  Args:
    signal (array): The sound signal (waveform) in the time domain. Should be
      flattened, i.e., the shape is (n_samples,).
    sr (int): Sampling rate associated with the signal waveform.
    n (int): Number of filters (subbands) to be generated with standard
      sampling (i.e., using a sampling factor of 1). Note, the actual number of
      filters in the generated filterbank depends on the sampling factor, and
      will also include lowpass and highpass filters that allow for
      perfect reconstruction of the input signal (the exact number of lowpass
      and highpass filters is determined by the sampling factor).
    low_lim (int): Lower limit of frequency range. Filters will not be defined
      below this limit.
    hi_lim (int): Upper limit of frequency range. Filters will not be defined
      above this limit.
    sample_factor (int): Positive integer that determines how densely ERB function
     will be sampled to create bandpass filters. 1 represents standard sampling;
     adjacent bandpass filters will overlap by 50%. 2 represents 2x overcomplete sampling;
     adjacent bandpass filters will overlap by 75%. 4 represents 4x overcomplete sampling;
     adjacent bandpass filters will overlap by 87.5%.
    padding_size (int, optional): If None (default), the signal will not be padded
      before filtering. Otherwise, the filters will be created assuming the
      waveform signal will be padded to length padding_size+signal_length.
    downsample (None, int, callable, optional): The `downsample` argument can
      be an integer representing the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor), a callable
      (to perform custom downsampling), or None to return the
      unmodified cochleagram; see `apply_envelope_downsample` for more
      information. If `ret_mode` is 'envs', this will be applied to the
      cochleagram before the nonlinearity, otherwise no downsampling will be
      performed. Providing a callable for custom downsampling is suggested.
    nonlinearity ({None, 'db', 'power', callable}, optional): The `nonlinearity`
      argument can be an predefined type, a callable
      (to apply a custom nonlinearity), or None to return the unmodified
      cochleagram; see `apply_envelope_nonlinearity` for more information.
      If `ret_mode` is 'envs', this will be applied to the cochleagram after
      downsampling, otherwise no nonlinearity will be applied. Providing a
      callable for applying a custom nonlinearity is suggested.
    fft_mode ({'auto', 'fftw', 'np'}, optional): Determine what implementation
      to use for FFT-like operations. 'auto' will attempt to use pyfftw, but
      will fallback to numpy, if necessary.
    ret_mode ({'envs', 'subband', 'analytic', 'all'}): Determines what will be
      returned. 'envs' (default) returns the subband envelopes; 'subband'
      returns just the subbands, 'analytic' returns the analytic signal provided
      by the Hilbert transform, 'all' returns all local variables created in this
      function.
    strict (bool, optional): If True (default), will include the extra
      highpass and lowpass filters required to make the filterbank invertible.
      If False, this will only perform calculations on the bandpass filters; note
      this decreases the number of frequency channels in the output by
       2 * `sample_factor`.
      function is used in a way that is unsupported by the MATLAB implemenation.
    strict (bool, optional): If True (default), will throw an errors if this
      function is used in a way that is unsupported by the MATLAB implemenation.
  Returns:
    array:
    **out**: The output, depending on the value of `ret_mode`. If the `ret_mode`
      is 'envs' and a downsampling and/or nonlinearity
      operation was requested, the output will reflect these operations.
  """
    if strict:
        if not isinstance(sr, int):
            raise ValueError("`sr` must be an int; ignore with `strict`=False")
        # make sure low_lim and hi_lim are int
        if not isinstance(low_lim, int):
            raise ValueError("`low_lim` must be an int; ignore with `strict`=False")
        if not isinstance(hi_lim, int):
            raise ValueError("`hi_lim` must be an int; ignore with `strict`=False")

    ret_mode = ret_mode.lower()
    if ret_mode == "all":
        ret_all_sb = True
    else:
        ret_all_sb = False

    # verify n is positive
    if n <= 0:
        raise ValueError("number of filters `n` must be positive; found: %s" % n)

    # allow for batch generation without creating filters everytime
    batch_signal = reshape_signal_batch(signal)  # (batch_dim, waveform_samples)

    # only make the filters once
    if kwargs.get("no_hp_lp_filts"):
        erb_kwargs = {"no_highpass": True, "no_lowpass": True}
    else:
        erb_kwargs = {}
    filts, hz_cutoffs, freqs = make_erb_cos_filters_nx(
        batch_signal.shape[1],
        sr,
        n,
        low_lim,
        hi_lim,
        sample_factor,
        padding_size=padding_size,
        full_filter=True,
        strict=strict,
        **erb_kwargs
    )

    freqs_to_plot = np.log10(freqs)

    is_batch = batch_signal.shape[0] > 1
    for i in range(batch_signal.shape[0]):

        temp_signal_flat = reshape_signal_canonical(batch_signal[i, ...])

        if ret_mode == "envs" or ret_mode == "all":
            temp_sb = generate_subband_envelopes_fast(
                temp_signal_flat,
                filts,
                padding_size=padding_size,
                fft_mode=fft_mode,
                debug_ret_all=ret_all_sb,
            )
        elif ret_mode == "subband":
            temp_sb = generate_subbands(
                temp_signal_flat,
                filts,
                padding_size=padding_size,
                fft_mode=fft_mode,
                debug_ret_all=ret_all_sb,
            )
        elif ret_mode == "analytic":
            temp_sb = generate_subbands(
                temp_signal_flat, filts, padding_size=padding_size, fft_mode=fft_mode
            )
        else:
            raise NotImplementedError("`ret_mode` is not supported.")

        if ret_mode == "envs":
            if downsample is None or callable(downsample):
                # downsample is None or callable
                temp_sb = apply_envelope_downsample(temp_sb, downsample)
            else:
                # interpret downsample as new sampling rate
                temp_sb = apply_envelope_downsample(temp_sb, "poly", sr, downsample)
            temp_sb = apply_envelope_nonlinearity(temp_sb, nonlinearity)

        if i == 0:
            sb_out = np.zeros(([batch_signal.shape[0]] + list(temp_sb.shape)))
        sb_out[i] = temp_sb

    sb_out = sb_out.squeeze()
    if ret_mode == "all":
        out_dict = {}
        # add all local variables to out_dict
        for k in dir():
            if k != "out_dict":
                out_dict[k] = locals()[k]
        return out_dict
    else:
        return sb_out


# # Compute the cochleagram.
# feat = cochleagram(
#     librosa.load("audio/fold1/193698-2-0-140.wav", sr=1000)[0],
#     n=8,
#     sr=1000,
#     low_lim=10,
#     hi_lim=20000,
#     sample_factor=2,
#     padding_size=None,
#     downsample=2,
#     nonlinearity=None,
#     fft_mode="auto",
#     ret_mode="envs",
#     strict=True,
# )
# feat = np.flipud(feat)
# # Plot the cochleagram.
# plt.figure(figsize=(10, 4))
# plt.imshow(
#     feat, aspect="auto", cmap="magma", origin="lower", interpolation="nearest"
# )
# plt.colorbar(format="%+2.0f dB")
# plt.title("Cochleagram")
# plt.tight_layout()
# plt.show()
