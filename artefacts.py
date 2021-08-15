import os
import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import pywt


my_path = os.path.dirname(__file__)

# =========================================================================== #
# DOC
#
# Module de gestion des artefacts qui correspond à la méthode 4 de l'ancien projet
# 
# Il y a plusieurs paramètres qui modifient le lissage de la courbe :

# Threshold
# Désigne le seuil à partir duquel les valeurs vont être modifiées.

# Paramètre 1 : threshold mode
# https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
# Soft:
# Data values with absolute value less than param are replaced with substitute. Data values with absolute value greater or equal to the thresholding value are shrunk toward zero by value. In other words, the new value is data/np.abs(data) * np.maximum(np.abs(data) - value, 0).

# Hard:
# The data values where their absolute value is less than the value param are replaced with substitute. Data values with absolute value greater or equal to the thresholding value stay untouched.

# Garrot:
# Corresponds to the Non-negative garrote threshold. It is intermediate between hard and soft thresholding. It behaves like soft thresholding for small data values and approaches hard thresholding for large data values.

# Greater:
# The data is replaced with substitute where data is below the thresholding value. Greater data values pass untouched.

# Less:
# The data is replaced with substitute where data is above the thresholding value. Lesser data values pass untouched.

# Paramètre 2 : wavelet mode
# https://www.mathworks.com/help/wavelet/gs/introduction-to-the-wavelet-families.html
# db - Daubechies
# The dbN wavelets are the Daubechies’ extremal phase wavelets. N refers to the number of vanishing moments. These filters are also referred to in the literature by the number of filter taps, which is 2N.

# Symlet Wavelets: symN
# The symN wavelets are also known as Daubechies’ least-asymmetric wavelets. The symlets are more symmetric than the extremal phase wavelets. In symN, N is the number of vanishing moments. These filters are also referred to in the literature by the number of filter taps, which is 2N.

# Coiflet Wavelets: coifN
# Coiflet scaling functions also exhibit vanishing moments. In coifN, N is the number of vanishing moments for both the wavelet and scaling functions. These filters are also referred to in the literature by the number of filter coefficients, which is 3N.

# dmey
# “Discrete” FIR approximation of Meyer wavelet, which is :

# Both ψ and φ are defined in the frequency domain, starting with an auxiliary function ν

# Biorthogonal Wavelet Pairs: biorNr.Nd
# Biorthogonal wavelets feature a pair of scaling functions and associated scaling filters — one for analysis and one for synthesis.

# There is also a pair of wavelets and associated wavelet filters — one for analysis and one for synthesis.

# The analysis and synthesis wavelets can have different numbers of vanishing moments and regularity properties. You can use the wavelet with the greater number of vanishing moments for analysis resulting in a sparse representation, while you use the smoother wavelet for reconstruction.

# rbio
# Reverse biorthogonal.

# Paramètre 3 : wavedec & waverec mode
# https://pywavelets.readthedocs.io/en/0.2.2/ref/signal-extension-modes.html#ref-modes
# zpd - zero-padding
# signal is extended by adding zero samples:

# ... 0 0 | x1 x2 ... xn | 0 0 ...

# cpd - constant-padding
# border values are replicated:

# ... x1 x1 | x1 x2 ... xn | xn xn ...

# sym - symmetric-padding
# signal is extended by mirroring samples:

# ... x2 x1 | x1 x2 ... xn | xn xn-1 ...

# ppd - periodic-padding
# signal is treated as a periodic one:

# ... xn-1 xn | x1 x2 ... xn | x1 x2 ...

# sp1 - smooth-padding
# signal is extended according to the first derivatives calculated on the edges (straight line)

# DWT
# performed for these extension modes is slightly redundant, but ensures perfect reconstruction. To receive the smallest possible number of coefficients, computations can be performed with the periodization mode:

# per - periodization
# is like periodic-padding but gives the smallest possible number of decomposition coefficients. IDWT must be performed with the same mode.
# 
# =========================================================================== #


threshold_slider = {'min':0, 'max':1, 'value':0.63}
threshold_mode = ['soft', 'hard', 'garrote', 'greater', 'less']
wavelet_mode = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'dmey', 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
waverdec_mode = ['zpd', 'cpd', 'sym', 'ppd', 'sp1', 'per']

data = pd.read_csv(os.path.join(my_path, "./cardio/42.csv"))

def lowpassfilter(df, signal, threshold, thresh_mode, wavelet_mode, wavedec_mode, waverec_mode):
    '''
    Fonction qui s'apparente à un filtre passe-bas
    '''
    data = df.copy()
    threshold = threshold*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet_mode, mode=wavedec_mode)
    coeff[1:] = (pywt.threshold(i, value=threshold, mode=thresh_mode) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet_mode, mode=waverec_mode)
    data['HR-method4'] = pd.DataFrame(reconstructed_signal, columns= {'HR-method4'})['HR-method4']
    return data

def smoothing(df, threshold_slider, threshold_mode, wavelet_mode, waverdec_mode):
    '''
    Fonction de lissage de la courbe en fonction des paramètres
    Retourne le dataframe d'origine avec la colonne de résultats
    '''
    wavedec_mode = waverdec_mode
    waverec_mode = waverdec_mode
    signal = df.HR.fillna(np.mean(df.HR)).values # Toutes les valeurs doivent exister
    rec = lowpassfilter(df, signal, threshold_slider, threshold_mode, wavelet_mode, wavedec_mode, waverec_mode)
    return rec

# print(smoothing(data, 0.63, 'soft', 'db8', 'per'))
