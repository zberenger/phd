import numpy as np

def interpol_cubic_irr(x, y, xi, sort=True, **kwargs):
    """
    1D cubic convolution interpolation on irregularly gridded input. Cython version is
    multithreaded and supports a 'threads' keyword to set the number of threads
    (default: # of processors)

    :author: Andreas Reigber
    :param x: The abscissa values
    :type y: 1-D ndarray float
    :param y: The ordinate values
    :type y: 1-D ndarray float, same length as x
    :param xi: The positions where the interpolates are desired
    :type xi: 1-D ndarray float

    :returns: The interpolated signal
    """

    if sort is True:
        sidx = np.argsort(x)
        x = x[sidx]
        y = y[sidx]

    n = len(y)
    yi = np.empty(xi.shape, y.dtype)
    for i in range(len(xi)):
        if xi[i] < x[0] or xi[i] > x[-1]:
            print('WARNING: Bad x input to cubiconv ==> 0 <= x <= len(y)')
            yi[i] = np.nan

        klo = 0
        khi = n-1
        while khi - klo > 1:
            k = int((khi + klo) / 2.0)
            if x[k] > xi[i]:
                khi = k
            else:
                klo = k
        h = x[khi] - x[klo]
        if h == 0.0:
            print('WARNING: Bad x input to cubiconv ==> x values must be distinct')
            yi[i] = np.nan

        kmi = klo - 1
        kpl = khi + 1

        if kmi < 0:
            a = 3 * y[klo] - 3 * y[khi] + y[kpl]
        else:
            a = y[kmi]

        if kpl > n-1:
            d = 3 * y[khi] - 3 * y[klo] + y[kmi]
        else:
            d = y[kpl]

        b = y[klo]
        c = y[khi]
        t = (xi[i] - x[klo]) / h
        t2 = t * t
        t3 = t2 * t
        h2 = h * h
        c00 = (- t3 + 2 * t2 - t) / 2.0
        c10 = (3 * t3 - 5 * t2 + 2) / 2.0
        c20 = (- 3 * t3 + 4 * t2 + t) / 2.0
        c30 = (t3 - t2) / 2.0
        yi[i] = a * c00 + b * c10 + c * c20 + d * c30
    return yi 
