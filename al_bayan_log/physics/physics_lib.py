# -*- coding: utf-8 -*-
# PhysiCAL Core - Python Port (v1.2)
# =================================================

import math

class Dimension:
    """
    Represents the 7 base SI dimensions: [L, M, T, I, Θ, N, J]
    """
    def __init__(self, L=0, M=0, T=0, I=0, Theta=0, N=0, J=0):
        self.dims = [L, M, T, I, Theta, N, J]

    def __eq__(self, other):
        return self.dims == other.dims

    def multiply(self, other):
        new_dims = []
        for i in range(7):
            new_dims.append(self.dims[i] + other.dims[i])
        return Dimension(*new_dims)

    def divide(self, other):
        new_dims = []
        for i in range(7):
            new_dims.append(self.dims[i] - other.dims[i])
        return Dimension(*new_dims)

    def power(self, n):
        new_dims = []
        for i in range(7):
            new_dims.append(self.dims[i] * n)
        return Dimension(*new_dims)

# Base Dimensions
DIM_SCALAR = Dimension()
DIM_LENGTH = Dimension(L=1)
DIM_MASS   = Dimension(M=1)
DIM_TIME   = Dimension(T=1)
DIM_CURRENT = Dimension(I=1)
DIM_TEMP   = Dimension(Theta=1)


class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5
    
    def __str__(self):
        return f"<{self.x:.2f}, {self.y:.2f}, {self.z:.2f}>"
    
    def __repr__(self):
        return self.__str__()


class PhysicalQuantity:
    """
    A value (Scalar or Vector) coupled with its physical dimensions and unit label.
    """
    def __init__(self, value, dimensions, unit_name="unit"):
        # Check if value is list/tuple -> convert to Vector3
        if isinstance(value, (list, tuple)):
             if len(value) == 3:
                 self.value = Vector3(float(value[0]), float(value[1]), float(value[2]))
             elif len(value) == 2:
                 self.value = Vector3(float(value[0]), float(value[1]), 0.0)
             else:
                 self.value = float(value[0])
        elif hasattr(value, 'x') and hasattr(value, 'magnitude'):
             self.value = value # It's already a Vector3
        else:
             self.value = float(value)
        
        self.dimensions = dimensions
        self.unit_name = unit_name
    
    @property
    def is_vector(self):
        return hasattr(self.value, 'magnitude')
    
    def magnitude(self):
        if self.is_vector:
            return self.value.magnitude()
        return abs(self.value)

    def __add__(self, other):
        if (isinstance(other, (int, float))):
            if other == 0: return self
            raise ValueError(f"❌ Dimensional Mismatch: Cannot add scalar {other} to {self.unit_name}")
        
        if (self.dimensions != other.dimensions):
            raise ValueError(f"❌ Dimensional Mismatch: Cannot add {self.unit_name} and {other.unit_name}")
        
        return PhysicalQuantity(self.value + other.value, self.dimensions, self.unit_name)

    def __sub__(self, other):
        if (isinstance(other, (int, float))):
            if other == 0: return self
            raise ValueError(f"❌ Dimensional Mismatch: Cannot subtract scalar {other} from {self.unit_name}")
        
        if (self.dimensions != other.dimensions):
            # Try to be lenient with compatible dims or just fail
            raise ValueError(f"❌ Dimensional Mismatch: Cannot subtract {self.unit_name} and {other.unit_name}")
        
        return PhysicalQuantity(self.value - other.value, self.dimensions, self.unit_name)

    def __mul__(self, other):
        if (isinstance(other, (int, float))):
            return PhysicalQuantity(self.value * other, self.dimensions, self.unit_name)
        
        # PQ * PQ
        new_dims = self.dimensions.multiply(other.dimensions)
        val = 0
        
        if self.is_vector and other.is_vector:
             # Assume Dot Product for F*d = Work? 
             # Or cross product for L = r x p?
             # For simplicity in this core, let's treat as component-wise or just fail if not specific
             # Wait, usually mul is dot product in 1D physics, but in 3D...
             # Let's fallback to scalar multiplication logic if one is scalar
             pass
        
        # Scalar logic (simplified)
        if self.is_vector and not hasattr(other.value, 'magnitude'):
             val = self.value * other.value
        elif not self.is_vector and hasattr(other.value, 'magnitude'):
             val = other.value * self.value
        else:
             # Scalar * Scalar or Vector*Vector (Undefined here, use explicit dot())
             # Actually, if both are scalar:
             if not self.is_vector and not other.is_vector:
                val = self.value * other.value
             else:
                # Placeholder for complex vector math
                val = self.value # ERROR Check needed
        
        return PhysicalQuantity(val, new_dims, f"({self.unit_name}*{other.unit_name})")

    def __truediv__(self, other):
        if (isinstance(other, (int, float))):
            return PhysicalQuantity(self.value / other, self.dimensions, self.unit_name)
        
        new_dims = self.dimensions.divide(other.dimensions)
        
        # Vector / Scalar
        if self.is_vector and not hasattr(other.value, 'magnitude'):
             val = self.value / other.value
             return PhysicalQuantity(val, new_dims, f"({self.unit_name}/{other.unit_name})")
        
        # Scalar / Scalar 
        if not self.is_vector and not hasattr(other.value, 'magnitude'):
             val = self.value / other.value
             return PhysicalQuantity(val, new_dims, f"({self.unit_name}/{other.unit_name})")
        
        raise ValueError("❌ Vector Division Undefined")

    def __str__(self):
        return f"{self.value} {self.unit_name}"

    def __repr__(self):
        return self.__str__()
    
    # Operators
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)


# --- Helper Factory Functions ---
def Meter(v): return PhysicalQuantity(v, DIM_LENGTH, "m") 
def Kg(v): return PhysicalQuantity(v, DIM_MASS, "kg") 
def Second(v): return PhysicalQuantity(v, DIM_TIME, "s") 

# Derived Helpers
def Velocity(v): return PhysicalQuantity(v, DIM_LENGTH.divide(DIM_TIME), "m/s") 

def Force(v): 
    # F = M * L / T^2
    dim_force = DIM_MASS.multiply(DIM_LENGTH).divide(DIM_TIME.power(2))
    return PhysicalQuantity(v, dim_force, "N")

def Acceleration(v): 
    # a = L / T^2
    dim_accel = DIM_LENGTH.divide(DIM_TIME.power(2))
    return PhysicalQuantity(v, dim_accel, "m/s^2")
