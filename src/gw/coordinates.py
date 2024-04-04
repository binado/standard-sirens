import numpy as np
from scipy.spatial.transform import Rotation as R


class Coordinates:
    def __init__(self, ndim) -> None:
        self.ndim = ndim


class CartesianCoordinates(Coordinates):
    def __init__(self, ndim, vector) -> None:
        super().__init__(ndim)
        self._v = vector

    @property
    def to_array(self):
        return self._v

    def __add__(self, cartesian_coordinates):
        """
        Coordinate addition
        """
        return CartesianCoordinates(self.ndim, self.to_array + cartesian_coordinates.to_array)

    def __sub__(self, cartesian_coordinates):
        """
        Coordinate subtraction
        """
        return CartesianCoordinates(self.ndim, self.to_array - cartesian_coordinates.to_array)

    def __mul__(self, cartesian_coordinates):
        """
        Coordinate dot-product
        """
        return np.dot(self.to_array, cartesian_coordinates.to_array)

    @property
    def norm(self):
        """
        Norm of coordinate vector
        """
        return np.sqrt(self * self)

    def cross(self, cartesian_coordinates):
        """
        Coordinate cross-product
        """
        return np.cross(self.to_array, cartesian_coordinates.to_array)

    def scalarmul(self, scalar: float):
        self._v *= scalar
        return self

    @classmethod
    def translate(cls, coordinates, a):
        """
        Coordinate translation by constant array
        """
        if len(a) != coordinates.ndim:
            raise ValueError("Arguments do not have the same dimension")
        return cls(coordinates.ndim, coordinates.to_array + a)


class Cartesian3DCoordinates(CartesianCoordinates):
    ndim = 3

    def __init__(self, vector) -> None:
        super().__init__(self.ndim, vector)

    @property
    def x(self):
        return self.to_array[0]

    @property
    def y(self):
        return self.to_array[1]

    @property
    def z(self):
        return self.to_array[-1]

    def rotate_frame(self, alpha: float, beta: float, gamma: float, **kwargs):
        """
        Return coordinates of the same vector in a rotated coordinate frame.
        alpha, beta and gamma describe the Euler angles necessary to rotate the current frame
        onto the target frame (e.g angular coordinates of detector in geocentric frame).

        Parameters
        ----------
        alpha: float
            Angle corresponding to the rotation around the Y-axis. Alias: theta
        beta: float
            Angle corresponding to the rotation around the first Z-axis. Alias: phi
        gamma: float
            Angle corresponding to the rotation around the second Z-axis. Alias: detector x-arm rotation

        Returns
        -------
        Cartesian3DCoordinates
            coordinates in rotated frame (e.g coordinates in detector frame)
        """
        rotation = R.from_euler("ZYZ", angles=[beta, alpha, gamma], **kwargs).inv()
        vector = rotation.apply(self.to_array)
        return self.__class__(vector)

    def rotate_vector(self, alpha: float, beta: float, gamma: float, **kwargs):
        """
        Return coordinates of a new, rotated vector in a fixed coordinate frame.
        alpha, beta and gamma describe the Euler angles necessary to rotate the target frame
        onto the input frame (e.g angular coordinates of detector in geocentric frame).

        Parameters
        ----------
        alpha: float
            Angle corresponding to the rotation around the Y-axis. Alias: theta
        beta: float
            Angle corresponding to the rotation around the first Z-axis. Alias: phi
        gamma: float
            Angle corresponding to the rotation around the second Z-axis. Alias: detector x-arm rotation

        Returns
        -------
        Cartesian3DCoordinates
            coordinates in rotated frame (e.g coordinates in gecentric frame)
        """
        rotation = R.from_euler("ZYZ", angles=[beta, alpha, gamma], **kwargs)
        vector = rotation.apply(self.to_array)
        return self.__class__(vector)


class Spherical3DCoordinates(Coordinates):
    ndim = 3

    def __init__(self, theta, phi, r=1.0) -> None:
        super().__init__(self.ndim)
        self.r = r
        self.theta = theta
        self.phi = phi

    @property
    def angles(self):
        return self.theta, self.phi

    def to_cartesian(self) -> Cartesian3DCoordinates:
        x = self.r * np.sin(self.theta) * np.cos(self.phi)
        y = self.r * np.sin(self.theta) * np.sin(self.phi)
        z = self.r * np.cos(self.theta)
        return Cartesian3DCoordinates(np.array([x, y, z]))

    @classmethod
    def from_cartesian(cls, coordinates: Cartesian3DCoordinates):
        x, y, z = coordinates.to_array
        r = coordinates.norm
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return cls(theta, phi, r)

    def __add__(self, spherical_coordinates):
        return self.from_cartesian(self.to_cartesian() + spherical_coordinates.to_cartesian())

    def __mul__(self, spherical_cordinates):
        return self.from_cartesian(self.to_cartesian() * spherical_cordinates.to_cartesian())
