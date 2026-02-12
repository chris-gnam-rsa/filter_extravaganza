import numpy as np

from rsalib import DATA_PATH
from rsalib.constants import M_TO_KM, MU_EARTH


def iau80in() -> tuple[np.array, np.array]:
    """Initialize the nutation matrices needed for reduction calculations.
    Converted to Python from https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/iau80in.m"""

    convrt = 0.0001 * np.pi / (180 * 3600.0)  # 0.0001" to rad

    nut80 = np.loadtxt(DATA_PATH / "nut80.dat")

    iar80 = nut80[:, 0:5]
    rar80 = nut80[:, 5:9]

    rar80[:106, :4] *= convrt

    return iar80, rar80


def truemean(
    ttt: float, order: int, eqeterms: int, opt: str
) -> tuple[float, float, float, float, float, np.array]:
    """Form the transformation matrix to go between the norad true equator
    mean equinox of date and the mean equator mean equinox of date (eci).
    The results approximate the effects of nutation and precession. Converted to Python from
    https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/truemean.m

    Args:
        ttt (float): Julian centuries of tt
        order (int): number of terms for nutation [4, 50, 106, ...]
        eqeterms (int): number of terms for eqe [0, 1, 2]
        opt (str): option for processing ['a' complete nutation, 'b' truncated nutation, 'c' truncated transf matrix]

    Returns:
        tuple[float, float, float, float, float, np.array]: 5 angles and the transformation matrix
    """

    deg2rad = np.pi / 180.0

    iar80, rar80 = iau80in()

    # Determine coefficients for iau 1980 nutation theory
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt
    ttt4 = ttt2 * ttt2

    meaneps = -46.8150 * ttt - 0.00059 * ttt2 + 0.001813 * ttt3 + 84381.448
    meaneps = np.fmod(meaneps / 3600.0, 360.0)
    meaneps *= deg2rad

    l_ = (
        134.96340251
        + (1717915923.2178 * ttt + 31.8792 * ttt2 + 0.051635 * ttt3 - 0.00024470 * ttt4)
        / 3600.0
    )
    l1 = (
        357.52910918
        + (129596581.0481 * ttt - 0.5532 * ttt2 - 0.000136 * ttt3 - 0.00001149 * ttt4)
        / 3600.0
    )
    f = (
        93.27209062
        + (1739527262.8478 * ttt - 12.7512 * ttt2 + 0.001037 * ttt3 + 0.00000417 * ttt4)
        / 3600.0
    )
    d = (
        297.85019547
        + (1602961601.2090 * ttt - 6.3706 * ttt2 + 0.006593 * ttt3 - 0.00003169 * ttt4)
        / 3600.0
    )
    omega = (
        125.04455501
        + (-6962890.2665 * ttt + 7.4722 * ttt2 + 0.007702 * ttt3 - 0.00005939 * ttt4)
        / 3600.0
    )

    l_ = np.fmod(l_, 360.0) * deg2rad
    l1 = np.fmod(l1, 360.0) * deg2rad
    f = np.fmod(f, 360.0) * deg2rad
    d = np.fmod(d, 360.0) * deg2rad
    omega = np.fmod(omega, 360.0) * deg2rad

    deltapsi = 0.0
    deltaeps = 0.0

    for i in range(0, order):  # the eqeterms in nut80.dat are already sorted
        tempval = (
            iar80[i, 0] * l_
            + iar80[i, 1] * l1
            + iar80[i, 2] * f
            + iar80[i, 3] * d
            + iar80[i, 4] * omega
        )
        deltapsi = deltapsi + (rar80[i, 0] + rar80[i, 1] * ttt) * np.sin(tempval)
        deltaeps = deltaeps + (rar80[i, 2] + rar80[i, 3] * ttt) * np.cos(tempval)

    # find nutation parameters
    deltapsi = np.fmod(deltapsi, 360.0) * deg2rad
    deltaeps = np.fmod(deltaeps, 360.0) * deg2rad
    trueeps = meaneps + deltaeps

    cospsi = np.cos(deltapsi)
    sinpsi = np.sin(deltapsi)
    coseps = np.cos(meaneps)
    sineps = np.sin(meaneps)
    costrueeps = np.cos(trueeps)
    sintrueeps = np.sin(trueeps)

    jdttt = ttt * 36525.0 + 2451545.0

    # small disconnect with ttt instead of ut1
    if (jdttt > 2450449.5) & (eqeterms > 0):
        eqe = (
            deltapsi * np.cos(meaneps)
            + 0.00264 * np.pi / (3600 * 180) * np.sin(omega)
            + 0.000063 * np.pi / (3600 * 180) * np.sin(2.0 * omega)
        )
    else:
        eqe = deltapsi * np.cos(meaneps)

    nut = np.zeros((3, 3))
    st = np.zeros((3, 3))
    nut[0, 0] = cospsi
    nut[0, 1] = costrueeps * sinpsi

    if opt == "b":
        nut[0, 1] = 0.0
    nut[0, 2] = sintrueeps * sinpsi
    nut[1, 0] = -coseps * sinpsi
    if opt == "b":
        nut[1, 0] = 0.0

    nut[1, 1] = costrueeps * coseps * cospsi + sintrueeps * sineps
    nut[1, 2] = sintrueeps * coseps * cospsi - sineps * costrueeps
    nut[2, 0] = -sineps * sinpsi
    nut[2, 1] = costrueeps * sineps * cospsi - sintrueeps * coseps
    nut[2, 2] = sintrueeps * sineps * cospsi + costrueeps * coseps

    st[0, 0] = np.cos(eqe)
    st[0, 1] = -np.sin(eqe)
    st[0, 2] = 0.0
    st[1, 0] = np.sin(eqe)
    st[1, 1] = np.cos(eqe)
    st[1, 2] = 0.0
    st[2, 0] = 0.0
    st[2, 1] = 0.0
    st[2, 2] = 1.0

    nutteme = nut * st

    if opt == "c":
        nutteme[0, 0] = 1.0
        nutteme[0, 1] = 0.0
        nutteme[0, 2] = deltapsi * sineps
        nutteme[1, 0] = 0.0
        nutteme[1, 1] = 1.0
        nutteme[1, 2] = deltaeps
        nutteme[2, 0] = -deltapsi * sineps
        nutteme[2, 1] = -deltaeps
        nutteme[2, 2] = 1.0

    return deltapsi, trueeps, meaneps, omega, eqe, nutteme


def precess(ttt: float, opt: str) -> tuple[np.array, float, float, float, float]:
    """
    Calculate the the transformation matrix that accounts for the effects of precession. Converted to Python from
    https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/precess.m

    Args:
        ttt (float): Julian centuries of tt
        opt (str): Precession model option ['01', '02', '96', '80']

    Returns:
        tuple[np.array, float, float, float, float]: transformation matrix, precession angles (psia, wa, ea, xa)
    """

    convrt = np.pi / (180.0 * 3600.0)  # " to rad
    ttt2 = ttt * ttt
    ttt3 = ttt2 * ttt

    #  fk4 b1950 precession angles
    if opt == "50":
        psia = 50.3708 + 0.0050 * ttt
        wa = 0.0  # not sure which one is which...
        ea = 84428.26 - 46.845 * ttt - 0.00059 * ttt2 + 0.00181 * ttt3
        xa = 0.1247 - 0.0188 * ttt

        zeta = 2304.9969 * ttt + 0.302 * ttt2 + 0.01808 * ttt3
        theta = 2004.2980 * ttt - 0.425936 * ttt2 - 0.0416 * ttt3
        z = 2304.9969 * ttt + 1.092999 * ttt2 + 0.0192 * ttt3

        # ttt is tropical centuries from 1950 36524.22 days
        prec = np.zeros((3, 3))
        prec[0, 0] = 1.0 - 2.9696e-4 * ttt2 - 1.3e-7 * ttt3
        prec[0, 1] = 2.234941e-2 * ttt + 6.76e-6 * ttt2 - 2.21e-6 * ttt3
        prec[0, 2] = 9.7169e-3 * ttt - 2.07e-6 * ttt2 - 9.6e-7 * ttt3
        prec[1, 0] = -prec[0, 1]
        prec[1, 1] = 1.0 - 2.4975e-4 * ttt2 - 1.5e-7 * ttt3
        prec[1, 2] = -1.0858e-4 * ttt2
        prec[2, 0] = -prec[0, 2]
        prec[2, 1] = prec[1, 2]
        prec[2, 2] = 1.0 - 4.721e-5 * ttt2

        # pass these back out for testing
        psia = zeta
        wa = theta
        ea = z

    # iau 76 precession angles
    else:
        if opt == "80":
            psia = 5038.7784 * ttt - 1.07259 * ttt2 - 0.001147 * ttt3  # "
            wa = 84381.448 + 0.05127 * ttt2 - 0.007726 * ttt3
            ea = 84381.448 - 46.8150 * ttt - 0.00059 * ttt2 + 0.001813 * ttt3
            xa = 10.5526 * ttt - 2.38064 * ttt2 - 0.001125 * ttt3
            zeta = 2306.2181 * ttt + 0.30188 * ttt2 + 0.017998 * ttt3  # "
            theta = 2004.3109 * ttt - 0.42665 * ttt2 - 0.041833 * ttt3
            z = 2306.2181 * ttt + 1.09468 * ttt2 + 0.018203 * ttt3
        else:  # iau 03 precession angles
            oblo = 84381.406  # "
            psia = (
                (
                    ((-0.0000000951 * ttt + 0.000132851) * ttt - 0.00114045) * ttt
                    - 1.0790069
                )
                * ttt
                + 5038.481507
            ) * ttt  # "
            wa = (
                (
                    ((0.0000003337 * ttt - 0.000000467) * ttt - 0.00772503) * ttt
                    + 0.0512623
                )
                * ttt
                - 0.025754
            ) * ttt + oblo
            ea = (
                (
                    ((-0.0000000434 * ttt - 0.000000576) * ttt + 0.00200340) * ttt
                    - 0.0001831
                )
                * ttt
                - 46.836769
            ) * ttt + oblo
            xa = (
                (
                    ((-0.0000000560 * ttt + 0.000170663) * ttt - 0.00121197) * ttt
                    - 2.3814292
                )
                * ttt
                + 10.556403
            ) * ttt

            zeta = (
                (
                    ((-0.0000003173 * ttt - 0.000005971) * ttt + 0.01801828) * ttt
                    + 0.2988499
                )
                * ttt
                + 2306.083227
            ) * ttt + 2.650545  # "
            theta = (
                (
                    ((-0.0000001274 * ttt - 0.000007089) * ttt - 0.04182264) * ttt
                    - 0.4294934
                )
                * ttt
                + 2004.191903
            ) * ttt
            z = (
                (
                    ((0.0000002904 * ttt - 0.000028596) * ttt + 0.01826837) * ttt
                    + 1.0927348
                )
                * ttt
                + 2306.077181
            ) * ttt - 2.650545

    # convert units to rad
    psia *= convrt
    wa *= convrt
    ea *= convrt
    xa *= convrt
    zeta *= convrt
    theta *= convrt
    z *= convrt

    if opt == "80":
        coszeta = np.cos(zeta)
        sinzeta = np.sin(zeta)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        cosz = np.cos(z)
        sinz = np.sin(z)

        # form matrix mod to j2000
        prec = np.zeros((3, 3))
        prec[0, 0] = coszeta * costheta * cosz - sinzeta * sinz
        prec[0, 1] = coszeta * costheta * sinz + sinzeta * cosz
        prec[0, 2] = coszeta * sintheta
        prec[1, 0] = -sinzeta * costheta * cosz - coszeta * sinz
        prec[1, 1] = -sinzeta * costheta * sinz + coszeta * cosz
        prec[1, 2] = -sinzeta * sintheta
        prec[2, 0] = -sintheta * cosz
        prec[2, 1] = -sintheta * sinz
        prec[2, 2] = costheta

    elif opt != "50":  # do rotations instead
        oblo = oblo * convrt  # " to rad
        a4 = np.array(
            [[np.cos(-xa), np.sin(-xa), 0], [-np.sin(-xa), np.cos(-xa), 0], [0, 0, 1]]
        )
        a5 = np.array(
            [[1, 0, 0], [0, np.cos(wa), np.sin(wa)], [0, -np.sin(wa), np.cos(wa)]]
        )
        a6 = np.array(
            [
                [np.cos(psia), np.sin(psia), 0],
                [-np.sin(psia), np.cos(psia), 0],
                [0, 0, 1],
            ]
        )
        a7 = np.array(
            [
                [1, 0, 0],
                [0, np.cos(-oblo), np.sin(-oblo)],
                [0, -np.sin(-oblo), np.cos(-oblo)],
            ]
        )
        prec = a7 @ a6 @ a5 @ a4

    return prec, psia, wa, ea, xa


def _newton_nu(ecc: float, nu: float) -> tuple[float, float]:
    """Solve Kepler's EQ to get eccentric and mean anomaly. Converted to Python from https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/newtonnu.m

    Args:
        ecc (float): eccentricity
        nu (float): true anomaly

    Returns:
        tuple[float, float]: eccentric anomaly, mean anomaly
    """

    e0 = 999999.9
    m = 999999.9
    small = 0.00000001

    if np.abs(ecc) < small:  # Circular
        m = nu
        e0 = nu
    else:  # Elliptical
        if ecc < 1.0 - small:
            sine = np.sqrt(1.0 - ecc**2) * np.sin(nu) / (1.0 + ecc * np.cos(nu))
            cose = (ecc + np.cos(nu)) / (1.0 + ecc * np.cos(nu))
            e0 = np.arctan2(sine, cose)
            m = e0 - ecc * np.sin(e0)
        else:
            if ecc > (1.0 + small):  # Hyperbolic
                if ecc > 1.0 and (np.abs(nu) + 0.00001 < np.pi - np.arccos(1.0 / ecc)):
                    sine = np.sqrt(ecc**2 - 1.0) * np.sin(nu) / (1.0 + ecc * np.cos(nu))
                    e0 = np.arcsinh(sine)
                    m = ecc * np.sinh(e0) - e0
            else:  # Parabolic
                if np.abs(nu) < 168.0 * np.pi / 180.0:
                    e0 = np.tan(nu * 0.5)
                    m = e0 + (e0**3) / 3.0

    if ecc < 1.0:
        m = np.remainder(m, 2.0 * np.pi)
        if m < 0.0:
            m += 2.0 * np.pi
        e0 = np.remainder(e0, 2.0 * np.pi)

    return e0, m


def _angl_vallado(vec1: np.array, vec2: np.array) -> float:
    """Vallado's angl() function from # https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/angl.m
    Kept here for consistency, but can likely be replaced with the numpy builtin in the future.

    Args:
        vec1 (np.array): input 3-vector 1
        vec2 (np.array): input 3-vector 2

    Returns:
        float: angle between the two vectors in radians
    """

    small = 0.00000001
    undefined = 999999.1

    magv1 = np.linalg.norm(vec1)
    magv2 = np.linalg.norm(vec2)

    if magv1 * magv2 > small**2:
        temp = np.dot(vec1, vec2) / (magv1 * magv2)
        if np.abs(temp) > 1.0:
            temp = np.sign(temp) * 1.0
        theta = np.arccos(temp)
    else:
        theta = undefined
    return theta


def rv2coe(
    r: np.array, v: np.array
) -> tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """Convert ECI state to classical orbital elements. Adapted from https://github.com/Spacecraft-Code/Vallado/blob/master/Matlab/rv2coe.m

    Args:
        r (np.array): ECI position [m]
        v (np.array): ECI velocity [m/s]

    Returns:
        tuple[float, float, float, float, float, float, float, float, float, float, float]:
        p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper
    """
    r *= M_TO_KM
    v *= M_TO_KM

    small = 1.0e-10
    infinite = 999999.9
    undefined = 999999.1

    muin = MU_EARTH * M_TO_KM**3  # km^3/s^2

    magr = np.linalg.norm(r)
    magv = np.linalg.norm(v)

    # Find h, n, and e vectors
    hbar = np.cross(r, v)
    magh = np.linalg.norm(hbar)
    nbar = np.zeros(3)
    if magh > small:
        nbar[0] = -hbar[1]
        nbar[1] = hbar[0]
        nbar[2] = 0.0
        magn = np.linalg.norm(nbar)
        c1 = magv**2 - muin / magr
        rdotv = np.dot(r, v)

        ebar = np.zeros(3)
        ebar = (c1 * r - rdotv * v) / muin
        ecc = np.linalg.norm(ebar)

        # Find a, e, and semi-latus rectum
        sme = magv * magv * 0.5 - muin / magr
        if np.abs(sme) > small:
            a = -muin / (2.0 * sme)
        else:
            a = infinite
        p = magh**2 / muin

        # Find inclination
        hk = hbar[2] / magh
        incl = np.arccos(hk)

        # Determine type of orbit for later use
        typeorbit = "ei"  # Elliptical, parabolic, hyperbolic inclined
        if ecc < small:
            if incl < small or np.abs(incl - np.pi) < small:
                typeorbit = "ce"  # Circular equatorial
            else:
                typeorbit = "ci"  # Circular inclined
        else:
            if incl < small or np.abs(incl - np.pi) < small:
                typeorbit = "ee"  # Elliptical, parabolic, hyperbolic equatorial

        # Find longitude of ascending node
        if magn > small:
            temp = nbar[0] / magn
            if np.abs(temp) > 1.0:
                temp = np.sign(temp)
            omega = np.arccos(temp)
            if nbar[1] < 0.0:
                omega = 2 * np.pi - omega
        else:
            omega = undefined

        # Find argument of perigee
        if typeorbit == "ei":
            argp = _angl_vallado(nbar, ebar)
            if ebar[2] < 0.0:
                argp = 2 * np.pi - argp
        else:
            argp = undefined

        # Find true anomaly at epoch
        if typeorbit[0] == "e":
            nu = _angl_vallado(ebar, r)
            if rdotv < 0.0:
                nu = 2 * np.pi - nu
        else:
            nu = undefined

        # Find argument of latitude - circular inclined
        if typeorbit == "ci":
            arglat = _angl_vallado(nbar, r)
            if r[2] < 0.0:
                arglat = 2 * np.pi - arglat
            m = arglat
        else:
            arglat = undefined

        # Find longitude of perigee - elliptical equatorial
        if ecc > small and typeorbit == "ee":
            temp = ebar[0] / ecc
            if np.abs(temp) > 1.0:
                temp = np.sign(temp)

            lonper = np.arccos(temp)
            if ebar[1] < 0.0:
                lonper = 2 * np.pi - lonper

            if incl > 0.5 * np.pi:
                lonper = 2 * np.pi - lonper

        else:
            lonper = undefined

        # Find true longitude - circular equatorial
        if magr > small and typeorbit == "ce":
            temp = r[0] / magr
            if np.abs(temp) > 1.0:
                temp = np.sign(temp)

            truelon = np.arccos(temp)
            if r[1] < 0.0:
                truelon = 2 * np.pi - truelon

            if incl > 0.5 * np.pi:
                truelon = 2 * np.pi - truelon

            m = truelon
        else:
            truelon = undefined

        # Find mean anomaly for all orbits
        if typeorbit[0] == "e":
            _, m = _newton_nu(ecc, nu)

    else:
        p = undefined
        a = undefined
        ecc = undefined
        incl = undefined
        omega = undefined
        argp = undefined
        nu = undefined
        m = undefined
        arglat = undefined
        truelon = undefined
        lonper = undefined

    return p, a, ecc, incl, omega, argp, nu, m, arglat, truelon, lonper
