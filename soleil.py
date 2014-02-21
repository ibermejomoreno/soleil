"""
Program: Soleil
Language: Python

    Copyright (C) 2014 Ivan Bermejo-Moreno

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


Description:
Numerical solver for particle-laden flow with radiation

Flow solver (based on Johan Larsson's Hybrid code):
- Compressible Navier-Stokes equations
- Finite-differences (orders 2 and 6) on regular
  (possibly stretched) grids (2 and 3D)
- Flux-based formulation
- Fluid: perfect gas

Particles:
- Lagrangian integration
- Coupling with flow (currently one-way)

Time stepping:
- Runge-Kutta 4
"""

from optparse import OptionParser
import logging
import os
import sys
import numpy
import xml.dom.minidom

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('soleil')

class Stencils:
    '''Stencils'''

    def __init__(self,
                 type,
                 order,
                 split):

        self.type = type
        self.order = order
        self.split = split

        if self.type == "central":
            if self.order == 2:
                self.stencilSize = 3
                self.interpolateCoeffs = numpy.array([0.0,0.5])
                self.firstDerivativeCoeffs = numpy.array([0.0,0.5])
                self.firstDerivativeFaceCoeffs = numpy.array([0.0,1.0])
                self.firstDerivativeConvolveCoeffs = \
                  numpy.append(+self.firstDerivativeCoeffs[::-1],
                               -self.firstDerivativeCoeffs[1:])
                self.firstDerivativeFaceConvolveCoeffs = \
                  numpy.append(+self.firstDerivativeFaceCoeffs[::-1],
                               -self.firstDerivativeFaceCoeffs[1:])
                self.firstDerivativeModifiedWaveNumber = 1.0
                self.secondDerivativeModifiedWaveNumber = 4.0
            elif self.order == 6:
                self.stencilSize = 7
                self.interpolateCoeffs = \
                  numpy.array([0.0,37.0/60,-8.0/60.0,1.0/60.0])
                self.firstDerivativeCoeffs = \
                  numpy.array([0.0,45.0/60.0,-9.0/60.0, 1.0/60.0])
                self.firstDerivativeFaceCoeffs = \
                  numpy.array([0.0,49.0/36.0,-5.0/36.0,1.0/90.0])
                self.firstDerivativeConvolveCoeffs = \
                  numpy.append(+self.firstDerivativeCoeffs[::-1],
                               -self.firstDerivativeCoeffs[1:])
                self.firstDerivativeFaceConvolveCoeffs = \
                  numpy.append(+self.firstDerivativeFaceCoeffs[::-1],
                               -self.firstDerivativeFaceCoeffs[1:])
                self.firstDerivativeModifiedWaveNumber = 1.59
                self.secondDerivativeModifiedWaveNumber = 6.04

            else:
                raise RuntimeError('Stencil of type %s and order %d not '
                                   'implemented', self.type, self.order)
        else:
            raise RuntimeError('Stencil of type %s not implemented', self.type)

   
class Fluid:
    '''Fluid properties'''

    def __init__(self,
                 gasConstant,
                 gamma,
                 dynamicViscosityRef,
                 dynamicViscosityTemperatureRef,
                 prandtl):

        self.gasConstant = gasConstant
        self.gamma = gamma
        self.gammaMinus1 = gamma - 1.0
        self.dynamicViscosityRef = dynamicViscosityRef
        self.dynamicViscosityTemperatureRef = dynamicViscosityTemperatureRef
        self.cv = gasConstant/(gamma-1)
        self.cp = gamma*self.cv
        self.prandtl = prandtl
        self.cpOverPrandtl = self.cp / self.prandtl

    def GetSoundSpeed(self,temperature):
        # Return sound speed from formula: c = gamma * R * T
        return numpy.sqrt(self.gamma * self.gasConstant * temperature)

    def GetDynamicViscosity(self,temperature):
        # Return dynamic viscosity from formula:
        return self.dynamicViscosityRef * \
               ( temperature / self.dynamicViscosityTemperatureRef )**0.75

class Field:
    '''Field associated with a given mesh'''

    def __init__(self, mesh, dimensions):

        self.mesh = mesh
        self.dimensions = dimensions
        if self.dimensions == 1:
            self.data = numpy.empty([mesh.numPointsXWithGhost,
                                     mesh.numPointsYWithGhost,
                                     mesh.numPointsZWithGhost])
        else:
            self.data = numpy.empty([mesh.numPointsXWithGhost,
                                     mesh.numPointsYWithGhost,
                                     mesh.numPointsZWithGhost,
                                     dimensions])

    def Copy(self):
        """ Copy field contents and return new (independent) instance """
        # Create the instance
        destination = Field(self.mesh, self.dimensions)
        if self.dimensions == 1:
            destination.data[:,:,:] = self.data[:,:,:]
        else:
            destination.data[:,:,:,:] = self.data[:,:,:,:]

        return destination

    def CopyDataFrom(self,target):
        """ Copy field data contents from target"""
        if self.dimensions == 1:
            self.data[:,:,:] = target.data[:,:,:]
        else:
            self.data[:,:,:,:] = target.data[:,:,:,:]

    def SetToConstant(self,constantValue):
        """ Init all elements of field to a specified constant value"""
        if self.dimensions == 1:
            self.data[:,:,:] = constantValue
        else:
            self.data[:,:,:,:] = constantValue

    def UpdateGhost(self):
        """ Update ghost values in field"""
        if self.dimensions == 1:

            # In X
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsX < self.mesh.numGhostX:
                for idx in range(0,self.mesh.startXdx):
                    self.data[idx,:,:] = \
                      self.data[self.mesh.startXdx,:,:]
                for idx in range(self.mesh.startXdx+1,self.mesh.numPointsXWithGhost):
                    self.data[idx,:,:] = \
                      self.data[self.mesh.startXdx,:,:]
            # Update
            self.data[0:self.mesh.startXdx,:,:] = \
              self.data[self.mesh.endXdx-self.mesh.numGhostX:\
                        self.mesh.endXdx,:,:]
            self.data[self.mesh.endXdx:self.mesh.numPointsXWithGhost,:,:] = \
              self.data[self.mesh.startXdx:\
                        self.mesh.startXdx+self.mesh.numGhostX,:,:]

            # In Y
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsY < self.mesh.numGhostY:
                for jdx in range(0,self.mesh.startYdx):
                    self.data[:,jdx,:] = \
                      self.data[:,self.mesh.startYdx,:]
                for jdx in range(self.mesh.startYdx+1,self.mesh.numPointsYWithGhost):
                    self.data[:,jdx,:] = \
                      self.data[:,self.mesh.startYdx,:]
            # Update
            self.data[:,0:self.mesh.startYdx,:] = \
              self.data[:,self.mesh.endYdx-self.mesh.numGhostY:\
                          self.mesh.endYdx,:]
            self.data[:,self.mesh.endYdx:self.mesh.numPointsYWithGhost,:] = \
              self.data[:,self.mesh.startYdx:\
                          self.mesh.startYdx+self.mesh.numGhostY,:]

            # In Z
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsZ < self.mesh.numGhostZ:
                for kdx in range(0,self.mesh.startZdx):
                    self.data[:,:,kdx] = \
                      self.data[:,:,self.mesh.startZdx]
                for kdx in range(self.mesh.startZdx+1,self.mesh.numPointsZWithGhost):
                    self.data[:,:,kdx] = \
                      self.data[:,:,self.mesh.startZdx]
            # Update
            self.data[:,:,0:self.mesh.startZdx] = \
              self.data[:,:,self.mesh.endZdx-self.mesh.numGhostZ:\
                            self.mesh.endZdx]
            self.data[:,:,self.mesh.endZdx:self.mesh.numPointsZWithGhost] = \
              self.data[:,:,self.mesh.startZdx:\
                            self.mesh.startZdx+self.mesh.numGhostZ]
        else:

            # In X
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsX < self.mesh.numGhostX:
                for idx in range(0,self.mesh.startXdx):
                    self.data[idx,:,:,:] = \
                      self.data[self.mesh.startXdx,:,:,:]
                for idx in range(self.mesh.startXdx+1,self.mesh.numPointsXWithGhost):
                    self.data[idx,:,:,:] = \
                      self.data[self.mesh.startXdx,:,:,:]
            # Update
            self.data[0:self.mesh.startXdx,:,:,:] = \
              self.data[self.mesh.endXdx-self.mesh.numGhostX:\
                        self.mesh.endXdx,:,:,:]
            self.data[self.mesh.endXdx:self.mesh.numPointsXWithGhost,:,:,:] = \
              self.data[self.mesh.startXdx:\
                        self.mesh.startXdx+self.mesh.numGhostX,:,:,:]

            # In Y
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsY < self.mesh.numGhostY:
                for jdx in range(0,self.mesh.startYdx):
                    self.data[:,jdx,:,:] = \
                      self.data[:,self.mesh.startYdx,:,:]
                for jdx in range(self.mesh.startYdx+1,self.mesh.numPointsYWithGhost):
                    self.data[:,jdx,:,:] = \
                      self.data[:,self.mesh.startYdx,:,:]
            # Update
            self.data[:,0:self.mesh.startYdx,:,:] = \
              self.data[:,self.mesh.endYdx-self.mesh.numGhostY:\
                          self.mesh.endYdx,:,:]
            self.data[:,self.mesh.endYdx:self.mesh.numPointsYWithGhost,:,:] = \
              self.data[:,self.mesh.startYdx:\
                          self.mesh.startYdx+self.mesh.numGhostY,:,:]

            # In Z
            # Inactive coordinate: fill right ghosts before updating
            if self.mesh.numPointsZ < self.mesh.numGhostZ:
                for kdx in range(0,self.mesh.startZdx):
                    self.data[:,:,kdx,:] = \
                      self.data[:,:,self.mesh.startZdx,:]
                for kdx in range(self.mesh.startZdx+1,self.mesh.numPointsZWithGhost):
                    self.data[:,:,kdx,:] = \
                      self.data[:,:,self.mesh.startZdx,:]
            # Update
            self.data[:,:,0:self.mesh.startZdx,:] = \
              self.data[:,:,self.mesh.endZdx-self.mesh.numGhostZ:\
                            self.mesh.endZdx,:]
            self.data[:,:,self.mesh.endZdx:self.mesh.numPointsZWithGhost,:] = \
              self.data[:,:,self.mesh.startZdx:\
                            self.mesh.startZdx+self.mesh.numGhostZ,:]

    def GetInterior(self):
        """ Returns a numpy array with values of field at the interior points
            (i.e., excluding ghosts)"""
        if self.dimensions == 1:
            return self.data[self.mesh.startXdx:self.mesh.endXdx,
                             self.mesh.startXdx:self.mesh.endXdx,
                             self.mesh.startXdx:self.mesh.endXdx]
        else:
            return self.data[self.mesh.startXdx:self.mesh.endXdx,
                             self.mesh.startXdx:self.mesh.endXdx,
                             self.mesh.startXdx:self.mesh.endXdx, 
                             :]

    def GradientX(self, gradientX):
        """ Updates gradientX.data with the first derivative of the field
            along the X coordinate at the interior points
            (i.e., excluding ghosts)"""
        if self.dimensions == 1:
            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                    gradientX.data[self.mesh.startXdx:self.mesh.endXdx,\
                                   jdx,kdx] = \
                      numpy.convolve(self.data[:,jdx,kdx],
                        self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                        mode='valid')/\
                      self.mesh.dX[self.mesh.startXdx:self.mesh.endXdx]
        else:
            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                    for ddx in range(self.dimensions):
                        gradientX.data[self.mesh.startXdx:self.mesh.endXdx,\
                                       jdx,kdx,ddx] = \
                          numpy.convolve(self.data[:,jdx,kdx,ddx],
                            self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                                         mode='valid')/\
                          self.mesh.dX[self.mesh.startXdx:self.mesh.endXdx]

    def GradientY(self,gradientY):
        """ Updates gradientY.data with the first derivative of the field
            along the Y coordinate at the interior points
            (i.e., excluding ghosts)"""
        if self.dimensions == 1:
            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    gradientY.data[idx,self.mesh.startYdx:self.mesh.endYdx,\
                                   kdx] = \
                      numpy.convolve(self.data[idx,:,kdx],
                        self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                        mode='valid')/\
                      self.mesh.dY[self.mesh.startYdx:self.mesh.endYdx]
        else:
            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    for ddx in range(self.dimensions):
                        gradientY.data[idx,self.mesh.startYdx:self.mesh.endYdx,\
                                       kdx,ddx] = \
                          numpy.convolve(self.data[idx,:,kdx,ddx],
                            self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                            mode='valid')/\
                          self.mesh.dY[self.mesh.startYdx:self.mesh.endYdx]

    def GradientZ(self,gradientZ):
        """ Updates gradientZ.data with the first derivative of the field
            along the Z coordinate at the interior points
            (i.e., excluding ghosts)"""
        if self.dimensions == 1:
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    gradientZ.data[idx,jdx,\
                                   self.mesh.startZdx:self.mesh.endZdx] = \
                      numpy.convolve(self.data[idx,jdx,:],
                        self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                        mode='valid')/\
                      self.mesh.dZ[self.mesh.startZdx:self.mesh.endZdx]
        else:
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    for ddx in range(self.dimensions):
                        gradientZ.data[idx,jdx,\
                                       self.mesh.startZdx:self.mesh.endZdx,\
                                       ddx] = \
                          numpy.convolve(self.data[idx,jdx,:,ddx],
                            self.mesh.spatialStencils.firstDerivativeConvolveCoeffs,
                            mode='valid')/\
                          self.mesh.dZ[self.mesh.startZdx:self.mesh.endZdx]

#    def HessianX(self,gradientX,hessianX):
#        """ Updates hessianX.data with the second derivative of the field
#            along the X coordinate at the interior points
#            (i.e., excluding ghosts)"""
#        if self.dimensions == 1:
#            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
#                for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
#                    hessianX.data[self.mesh.startXdx:self.mesh.endXdx,\
#                                  jdx,kdx] = \
#                      (numpy.convolve(self.data[:,jdx,kdx],
#                         self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                         mode='valid') - \
#                       gradientX.data[self.mesh.startXdx,self.mesh.endXdx,\
#                                      jdx,kdx,ddx]*\
#                       self.mesh.d2X[self.mesh.startXdx:self.mesh.endXdx])/\
#                      self.mesh.dX[self.mesh.startXdx:self.mesh.endXdx]**2
#        else:
#            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
#                for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
#                    for ddx in range(self.dimensions):
#                        hessianX.data[self.mesh.startXdx:self.mesh.endXdx,\
#                                      jdx,kdx,ddx] = \
#                          (numpy.convolve(self.data[:,jdx,kdx,ddx],
#                             self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                             mode='valid') - \
#                           gradientX.data[self.mesh.startXdx:self.mesh.endXdx,\
#                                          jdx,kdx,ddx]*\
#                           self.mesh.d2X[self.mesh.startXdx:self.mesh.endXdx])/\
#                          self.mesh.dX[self.mesh.startXdx:self.mesh.endXdx]**2
#
#    def HessianY(self,gradientY,hessianY):
#        """ Updates hessianY.data with the second derivative of the field
#            along the Y coordinate at the interior points
#            (i.e., excluding ghosts)"""
#        if self.dimensions == 1:
#            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
#                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
#                    hessianY.data[idx,self.mesh.startYdx:self.mesh.endYdx,\
#                                  kdx] = \
#                      (numpy.convolve(self.data[idx,:,kdx],
#                         self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                         mode='valid') - \
#                       gradientY.data[idx,self.mesh.startYdx,self.mesh.endYdx,\
#                                      kdx,ddx]*\
#                       self.mesh.d2Y[self.mesh.startYdx:self.mesh.endYdx])/\
#                      self.mesh.dY[self.mesh.startYdx:self.mesh.endYdx]**2
#        else:
#            for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
#                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
#                    for ddx in range(self.dimensions):
#                        hessianY.data[idx,self.mesh.startYdx:self.mesh.endYdx,\
#                                      kdx,ddx] = \
#                          (numpy.convolve(self.data[idx,:,kdx,ddx],
#                             self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                             mode='valid') - \
#                           gradientY.data[idx,
#                                          self.mesh.startYdx:self.mesh.endYdx,\
#                                          kdx,ddx]*\
#                           self.mesh.d2Y[self.mesh.startYdx:self.mesh.endYdx])/\
#                      self.mesh.dY[self.mesh.startYdx:self.mesh.endYdx]**2
#
#    def HessianZ(self,gradientZ):
#        """ Updates hessianZ.data with the second derivative of the field
#            along the Z coordinate at the interior points
#            (i.e., excluding ghosts)"""
#        if self.dimensions == 1:
#            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
#                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
#                    hessianZ.data[idx,jdx,
#                                  self.mesh.startZdx:self.mesh.endZdx] = \
#                      (numpy.convolve(self.data[idx,jdx,:],
#                         self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                         mode='valid') - \
#                       gradientZ.data[idx,jdx,
#                                      self.mesh.startZdx,self.mesh.endZdx,\
#                                      ddx]*\
#                       self.mesh.d2Z[self.mesh.startZdx:self.mesh.endZdx])/\
#                      self.mesh.dZ[self.mesh.startZdx:self.mesh.endZdx]**2
#        else:
#            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
#                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
#                    for ddx in range(self.dimensions):
#                        hessianZ.data[idx,jdx,
#                                      self.mesh.startZdx:self.mesh.endZdx,\
#                                      ddx] = \
#                          (numpy.convolve(self.data[idx,jdx,:,ddx],
#                             self.mesh.spatialStencils.secondDerivativeConvolveCoeffs,
#                             mode='valid') - \
#                           gradientZ.data[idx,jdx,
#                                          self.mesh.startZdx:self.mesh.endZdx,\
#                                          ddx]*\
#                           self.mesh.d2Z[:])/\
#                      self.mesh.dZ[self.mesh.startZdx:self.mesh.endZdx]**2

    def WriteSlice(self,normalDirection,sliceIndex,outputFileNamePrefix,
                   includeGhost=False):

        if normalDirection == 0:
            normalDirectionString = "X"
            coorM = self.mesh.coorYWithGhost
            coorN = self.mesh.coorZWithGhost
            if includeGhost:
                minMdx = 0
                maxMdx = self.mesh.numPointsYWithGhost
                minNdx = 0
                maxNdx = self.mesh.numPointsZWithGhost
            else:
                minMdx = self.mesh.startYdx
                maxMdx = self.mesh.endYdx
                minNdx = self.mesh.startZdx
                maxNdx = self.mesh.endZdx
            if self.dimensions == 1:
                outputArray = self.data[sliceIndex,:,:]
            else:
                outputArray = self.data[sliceIndex,:,:,:]
        elif normalDirection == 1:
            normalDirectionString = "Y"
            coorM = self.mesh.coorXWithGhost
            coorN = self.mesh.coorZWithGhost
            if includeGhost:
                minMdx = 0
                maxMdx = self.mesh.numPointsXWithGhost
                minNdx = 0
                maxNdx = self.mesh.numPointsZWithGhost
            else:
                minMdx = self.mesh.startXdx
                maxMdx = self.mesh.endXdx
                minNdx = self.mesh.startZdx
                maxNdx = self.mesh.endZdx
            
            if self.dimensions == 1:
                outputArray = self.data[:,sliceIndex,:]
            else:
                outputArray = self.data[:,sliceIndex,:,:]
        elif normalDirection == 2:
            normalDirectionString = "Z"
            coorM = self.mesh.coorXWithGhost
            coorN = self.mesh.coorYWithGhost
            if includeGhost:
                minMdx = 0
                maxMdx = self.mesh.numPointsXWithGhost
                minNdx = 0
                maxNdx = self.mesh.numPointsYWithGhost
            else:
                minMdx = self.mesh.startXdx
                maxMdx = self.mesh.endXdx
                minNdx = self.mesh.startYdx
                maxNdx = self.mesh.endYdx
            if self.dimensions == 1:
                outputArray = self.data[:,:,sliceIndex]
            else:
                outputArray = self.data[:,:,sliceIndex,:]
        else:
            raise RuntimeError('Normal direction should be '
                               '0 (X), 1 (Y) or 2 (Z)')

        if self.dimensions == 1:
            outputFileName = outputFileNamePrefix \
                             + "_normalTo" + normalDirectionString \
                             + "_sliceAtIndex_" + str(sliceIndex) \
                             + ".txt"
            Write2DArrayToMatrix(coorM, coorN, 
                                 minMdx, maxMdx, 
                                 minNdx, maxNdx,
                                 outputArray,
                                 outputFileName)
        elif self.dimensions > 1:
            for dimension in range(self.dimensions):
                outputFileName = outputFileNamePrefix \
                                 + "_normalTo" + normalDirectionString \
                                 + "_sliceAtIndex_" + str(sliceIndex) \
                                 + "_dimension_" + str(dimension) + ".txt"
                Write2DArrayToMatrix(coorM, coorN, 
                                     minMdx, maxMdx, 
                                     minNdx, maxNdx,
                                     outputArray[:,:,dimension],
                                     outputFileName)


        else:
            raise RuntimeError('Writing not implemented')


class Flow:
    '''Flow class'''
 
    def __init__(self, inputFileName, mesh, fluid):
 
        log.info('Initializing flow')
 
        # Add reference to mesh and fluid properties
        self.mesh = mesh
        self.fluid = fluid
 
        # Init conserved variables
        self.rho = Field(mesh,1)
        self.rhoVel = Field(mesh,3)
        self.rhoEnergy = Field(mesh,1)
 
        # Init primitive variables
        self.velocity = Field(mesh,3)
        self.pressure = Field(mesh,1)
        self.temperature = Field(mesh,1)
 
        # Init gradients of primitive variables
        self.velocityGradientX = Field(mesh,3)
        self.velocityGradientY = Field(mesh,3)
        self.velocityGradientZ = Field(mesh,3)
 
        # Init sgs model
        self.sgsModelType = "none"
        self.sgsEnergy = Field(mesh,1)
        self.sgsEnergy.data.fill(0.0)
        self.sgsEddyViscosity = Field(mesh,1)
        self.sgsEddyViscosity.data.fill(0.0)
        self.sgsEddyKappa = Field(mesh,1)
        self.sgsEddyKappa.data.fill(0.0)
 
        # Read options from file
        dom = xml.dom.minidom.parse(inputFileName)
        flowElement = dom.getElementsByTagName("flow")[0]
        # Initial condition
        flowInitialConditionElement = \
          flowElement.getElementsByTagName("initialCondition")[0]
        flowInitialConditionType = \
          flowInitialConditionElement.attributes["type"].value
        flowInitialConditionValue = \
          flowInitialConditionElement.firstChild.data
        # Sgs model
        flowSgsModelElement = \
          flowElement.getElementsByTagName("sgsModel")[0]
        self.sgsModelType = \
          flowSgsModelElement.attributes["type"].value
    
        if flowInitialConditionType == "file":
            self.InitFromBinaryFile(flowInitialConditionValue)
            self.UpdatePrimitiveFromConserved()
            self.UpdateConservedFromPrimitive()
            self.UpdateGhost()
        elif flowInitialConditionType == "fileSlice":
            self.InitFromBinaryFileSlice(flowInitialConditionValue)
            self.UpdatePrimitiveFromConserved()
            self.UpdateConservedFromPrimitive()
            self.UpdateGhost()
        elif flowInitialConditionType == "dummy":
            self.InitDummy()
        elif flowInitialConditionType == "TaylorGreen":
            self.InitTaylorGreen()
            self.UpdateGhost()
        else:
            raise RuntimeError('Unknown type of flow initialization: %s',
                               flowInitialConditionType)

        self.UpdateAuxiliaryVelocity()
        self.UpdateAuxiliaryThermodynamics()
 
    def InitDummy(self):
 
        log.info('Initializing flow fields to dummy fields')
 
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    ldx = ( (idx-self.mesh.numGhostX) * \
                            self.mesh.numPointsY + \
                            (jdx-self.mesh.numGhostY) ) * \
                          self.mesh.numPointsZ + \
                          (kdx-self.mesh.numGhostZ)
                    self.rho.data[idx,jdx,kdx] = \
                        1.0
                    self.rhoEnergy.data[idx,jdx,kdx] = \
                        numpy.sin(self.mesh.coorXWithGhost[idx])
                    self.rhoVel.data[idx,jdx,kdx,0] = \
                        numpy.sin(self.mesh.coorYWithGhost[jdx])
                    self.rhoVel.data[idx,jdx,kdx,1] = \
                        numpy.sin(self.mesh.coorXWithGhost[idx])
                    self.rhoVel.data[idx,jdx,kdx,2] = \
                        numpy.sin(self.mesh.coorXWithGhost[idx])
 
    def InitTaylorGreen(self):
 
        log.info('Initializing flow fields to Taylor-Green vortex fields')

        taylorGreenPressure = 100.0
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    self.rho.data[idx,jdx,kdx] = \
                        1.0
                    self.rhoEnergy.data[idx,jdx,kdx] = \
                    self.velocity.data[idx,jdx,kdx,0] = \
                        numpy.sin(self.mesh.coorXWithGhost[idx]) * \
                        numpy.cos(self.mesh.coorYWithGhost[jdx]) * \
                        numpy.cos(self.mesh.coorZWithGhost[kdx])
                    self.velocity.data[idx,jdx,kdx,1] = \
                      - numpy.cos(self.mesh.coorXWithGhost[idx]) * \
                        numpy.sin(self.mesh.coorYWithGhost[jdx]) * \
                        numpy.cos(self.mesh.coorZWithGhost[kdx])
                    self.velocity.data[idx,jdx,kdx,2] = 0.0
                    factorA = numpy.cos(2.0*self.mesh.coorZWithGhost[kdx]) + \
                              2.0
                    factorB = numpy.cos(2.0*self.mesh.coorXWithGhost[idx]) + \
                              numpy.cos(2.0*self.mesh.coorYWithGhost[jdx])
                    self.pressure.data[idx,jdx,kdx] = \
                        taylorGreenPressure + (factorA*factorB - 2.0) / 16.0

        self.UpdateConservedFromPrimitive()
 
    def InitFromBinaryFile(self,inputFileName):
 
        log.info('Initializing flow fields from %s', inputFileName)
 
        try:
           import struct
           numVariables = 5
           with open(inputFileName, 'rb') as inputFile:
               # Header
               (precision, version, step,
                readNumPointsX, readNumPointsY, readNumPointsZ,
                numVariables) = struct.unpack('i'*7, inputFile.read(4*7))
               if readNumPointsX != self.mesh.numPointsX:
                   raise RuntimeError('Read number of points in X is %d != %d',
                                      readNumPointsX, self.mesh.numPointsX)
               if readNumPointsY != self.mesh.numPointsY:
                   raise RuntimeError('Read number of points in Y is %d != %d',
                                      readNumPointsY, self.mesh.numPointsY)
               if readNumPointsZ != self.mesh.numPointsZ:
                   raise RuntimeError('Read number of points in Z is %d != %d',
                                      readNumPointsZ, self.mesh.numPointsZ)
               if numVariables != 5:
                   raise RuntimeError('Number of variables in %s is %d',
                                      inputFileName, numVariables)
               variablesID = struct.unpack('i'*(numVariables), 
                                           inputFile.read(4*(numVariables)))
               (dummy,) = struct.unpack('d',inputFile.read(8))
               # Coordinates
               readCoorX = numpy.array(
                 struct.unpack('d'*self.mesh.numPointsX,
                               inputFile.read(8*self.mesh.numPointsX)))
               readCoorY = numpy.array(
                 struct.unpack('d'*self.mesh.numPointsY,
                               inputFile.read(8*self.mesh.numPointsY)))
               readCoorZ = numpy.array(
                 struct.unpack('d'*self.mesh.numPointsZ,
                               inputFile.read(8*self.mesh.numPointsZ)))
               # Compare coordinates with mesh coordinates
               errorX = readCoorX - self.mesh.coorX
               errorY = readCoorY - self.mesh.coorY
               errorZ = readCoorZ - self.mesh.coorZ
               if numpy.sqrt(errorX.dot(errorX)) > \
                  (readCoorX[1]-readCoorX[0])/1e6 or \
                  numpy.sqrt(errorY.dot(errorY)) > \
                  (readCoorY[1]-readCoorY[0])/1e6 or \
                  numpy.sqrt(errorZ.dot(errorZ)) > \
                  (readCoorZ[1]-readCoorZ[0])/1e6:
                   raise RuntimeError('Coor differ')
 
               # Fields
               numPoints = self.mesh.numPointsX * \
                           self.mesh.numPointsY * \
                           self.mesh.numPointsZ
               rho  = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhou = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhov = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhow = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhoe = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               # Assign read fields to flow fields
               for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                   for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                       for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                           ldx = ( (idx-self.mesh.numGhostX) * \
                                   self.mesh.numPointsY + \
                                   (jdx-self.mesh.numGhostY) ) * \
                                 self.mesh.numPointsZ + \
                                 (kdx-self.mesh.numGhostZ)
                           self.rho.data[idx,jdx,kdx] = rho[ldx]
                           self.rhoEnergy.data[idx,jdx,kdx] = rhoe[ldx]
                           self.rhoVel.data[idx,jdx,kdx,0] = rhou[ldx]
                           self.rhoVel.data[idx,jdx,kdx,1] = rhov[ldx]
                           self.rhoVel.data[idx,jdx,kdx,2] = rhow[ldx]
 
        except IOError:
           raise RuntimeError('Opening %s', inputFileName)
 
    def InitFromBinaryFileSlice(self,inputFileName):
 
        log.info('Initializing flow fields from %s', inputFileName)
 
        try:
           import struct
           numVariables = 5
           with open(inputFileName, 'rb') as inputFile:
               # Header
               (precision, version, step,
                readNumPointsX, readNumPointsY, readNumPointsZ,
                numVariables) = struct.unpack('i'*7, inputFile.read(4*7))
               if readNumPointsX != self.mesh.numPointsX:
                   raise RuntimeError('Read number of points in X is %d != %d',
                                      readNumPointsX, self.mesh.numPointsX)
               if readNumPointsY != self.mesh.numPointsY:
                   raise RuntimeError('Read number of points in Y is %d != %d',
                                      readNumPointsY, self.mesh.numPointsY)
               if 1 != self.mesh.numPointsZ:
                   raise RuntimeError('Number of points in Z should be 1')
               if numVariables != 5:
                   raise RuntimeError('Number of variables in %s is %d',
                                      inputFileName, numVariables)
               variablesID = struct.unpack('i'*(numVariables), 
                                           inputFile.read(4*(numVariables)))
               (dummy,) = struct.unpack('d',inputFile.read(8))
               # Coordinates
               readCoorX = numpy.array(
                 struct.unpack('d'*self.mesh.numPointsX,
                               inputFile.read(8*self.mesh.numPointsX)))
               readCoorY = numpy.array(
                 struct.unpack('d'*self.mesh.numPointsY,
                               inputFile.read(8*self.mesh.numPointsY)))
               readCoorZ = numpy.array(
                 struct.unpack('d'*readNumPointsZ,
                               inputFile.read(8*readNumPointsZ)))
               # Compare coordinates with mesh coordinates
               errorX = readCoorX - self.mesh.coorX
               errorY = readCoorY - self.mesh.coorY
               if numpy.sqrt(errorX.dot(errorX)) > \
                  (readCoorX[1]-readCoorX[0])/1e6 or \
                  numpy.sqrt(errorY.dot(errorY)) > \
                  (readCoorY[1]-readCoorY[0])/1e6:
                   raise RuntimeError('Coor differ')
 
               # Fields
               numPoints = self.mesh.numPointsX * \
                           self.mesh.numPointsY * \
                           readNumPointsZ
               rho  = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhou = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhov = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhow = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               rhoe = numpy.array(struct.unpack('f'*numPoints, 
                                                inputFile.read(4*numPoints)))
               # Assign read fields to flow fields
               for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                   for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                       for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                           ldx = ( (idx-self.mesh.numGhostX) * \
                                   self.mesh.numPointsY + \
                                   (jdx-self.mesh.numGhostY) ) * \
                                 readNumPointsZ + \
                                 (kdx-self.mesh.numGhostZ)
                           self.rho.data[idx,jdx,kdx] = rho[ldx]
                           self.rhoEnergy.data[idx,jdx,kdx] = rhoe[ldx]
                           self.rhoVel.data[idx,jdx,kdx,0] = rhou[ldx]
                           self.rhoVel.data[idx,jdx,kdx,1] = rhov[ldx]
                           self.rhoVel.data[idx,jdx,kdx,2] = 0.0
 
        except IOError:
           raise RuntimeError('Opening %s', inputFileName)
 
    def UpdateGhost(self):
 
        #log.info('Updating flow fields ghost data')
 
        self.rho.UpdateGhost()
        self.rhoVel.UpdateGhost()
        self.rhoEnergy.UpdateGhost()
        self.velocity.UpdateGhost()
        self.pressure.UpdateGhost()
        self.temperature.UpdateGhost()
 
    def UpdateGhostVelocityGradient(self):
 
        #log.info('Updating velocity gradient flow fields ghost data')
 
        self.velocityGradientX.UpdateGhost()
        self.velocityGradientY.UpdateGhost()
        self.velocityGradientZ.UpdateGhost()
 
    def UpdatePrimitiveFromConserved(self):
 
        #log.info('Updating primitive variables from conserved ones')
 
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    self.velocity.data[idx,jdx,kdx,:] = \
                      self.rhoVel.data[idx,jdx,kdx,:]/\
                      self.rho.data[idx,jdx,kdx]
                    # e = cv * T = E - kineticEnergy - sgsEnergy -> T
                    self.temperature.data[idx,jdx,kdx] = \
                      (self.rhoEnergy.data[idx,jdx,kdx]/\
                       self.rho.data[idx,jdx,kdx] - \
                       numpy.dot(self.velocity.data[idx,jdx,kdx,:],
                                 self.velocity.data[idx,jdx,kdx,:])/2.0 - \
                       self.sgsEnergy.data[idx,jdx,kdx]) / \
                      self.fluid.cv
                    # Equation of state: p = R * rho * T
                    self.pressure.data[idx,jdx,kdx] = \
                      self.fluid.gasConstant * \
                      self.rho.data[idx,jdx,kdx] * \
                      self.temperature.data[idx,jdx,kdx]
 
    def UpdateConservedFromPrimitive(self):
 
        #log.info('Updating conserved variables from primitive ones')
 
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
 
                    self.rhoVel.data[idx,jdx,kdx,:] = \
                      self.rho.data[idx,jdx,kdx] * \
                      self.velocity.data[idx,jdx,kdx,:]
 
                    # Equation of state: T = p / ( R * rho )
                    tmpTemperature = \
                      self.pressure.data[idx,jdx,kdx] / \
                      (self.fluid.gasConstant * \
                      self.rho.data[idx,jdx,kdx])
 
                    # rhoE = rhoe (= rho * cv * T) + kineticEnergy + sgsEnergy
                    self.rhoEnergy.data[idx,jdx,kdx] = \
                      self.rho.data[idx,jdx,kdx] * \
                      ( self.fluid.cv * tmpTemperature  \
                        + numpy.dot(self.velocity.data[idx,jdx,kdx,:],
                                    self.velocity.data[idx,jdx,kdx,:])/2.0 ) \
                      + self.sgsEnergy.data[idx,jdx,kdx]

    def InterpolateCentralX(self, idx, jdx, kdx,
                            rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy):
        '''Interpolate using central scheme to obtain fluxes along X'''

        rhoFactorDiagonal = 0.0
        rhoVelFactorDiagonal = numpy.zeros(3)
        rhoEnergyFactorDiagonal = 0.0
        fpdiag = 0.0

        directionIdx = 0
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
            coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
            idxPlus  = idx + ndx
            idxMinus = idx + 1 - ndx  # Note the +1 for the faces
            rhoFactorDiagonal += coeff * \
                      ( self.rho.data[idxMinus,jdx,kdx] * \
                        self.velocity.data[idxMinus,jdx,kdx,directionIdx] + \
                        self.rho.data[idxPlus,jdx,kdx] * \
                        self.velocity.data[idxPlus,jdx,kdx,directionIdx] )
            rhoVelFactorDiagonal[:] += coeff * \
                      ( self.rhoVel.data[idxMinus,jdx,kdx,:] * \
                        self.velocity.data[idxMinus,jdx,kdx,directionIdx] + \
                        self.rhoVel.data[idxPlus,jdx,kdx,:] * \
                        self.velocity.data[idxPlus,jdx,kdx,directionIdx] )
            rhoEnergyFactorDiagonal += coeff * \
                      ( rhoEnthalpy.data[idxMinus,jdx,kdx] * \
                        self.velocity.data[idxMinus,jdx,kdx,directionIdx] + \
                        rhoEnthalpy.data[idxPlus,jdx,kdx] * \
                        self.velocity.data[idxPlus,jdx,kdx,directionIdx] )
            fpdiag += coeff * \
                      ( self.pressure.data[idxMinus,jdx,kdx] + \
                        self.pressure.data[idxPlus,jdx,kdx] )
        # Bilinear fluxes
        rhoFactorSkew = 0.0
        rhoVelFactorSkew = numpy.zeros(3)
        rhoEnergyFactorSkew = 0.0

        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeCoeffs)
        #  mdx = -N+1,...,0
        for mdx in range(2-numberCoeffs,1):
          tmp = 0.0
          idxPlus = idx + mdx
          for ndx in range(1,mdx+numberCoeffs):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[ndx-mdx] * \
                   self.velocity.data[idx+ndx,jdx,kdx,directionIdx]

          rhoFactorSkew += self.rho.data[idxPlus,jdx,kdx] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idxPlus,jdx,kdx,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idxPlus,jdx,kdx] * tmp
        #  mdx = 1,...,N
        for mdx in range(1,numberCoeffs):
          tmp = 0.0
          idxPlus = idx + mdx
          for ndx in range(mdx-numberCoeffs+1,1):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[mdx-ndx] * \
                   self.velocity.data[idx+ndx,jdx,kdx,directionIdx]

          rhoFactorSkew += self.rho.data[idxPlus,jdx,kdx] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idxPlus,jdx,kdx,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idxPlus,jdx,kdx] * tmp

        #  Final split fluxes
        splitParameter = self.mesh.spatialStencils.split
        oneMinusSplitParameter = 1.0 - splitParameter
        rhoFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoFactorDiagonal + \
          oneMinusSplitParameter * rhoFactorSkew
        rhoVelFlux.data[idx,jdx,kdx,:] = \
          splitParameter * rhoVelFactorDiagonal[:] + \
          oneMinusSplitParameter * rhoVelFactorSkew[:]
        rhoEnergyFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoEnergyFactorDiagonal + \
          oneMinusSplitParameter * rhoEnergyFactorSkew

        rhoVelFlux.data[idx,jdx,kdx,directionIdx] += fpdiag

    def InterpolateCentralY(self, idx, jdx, kdx,
                            rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy):
        '''Interpolate using central scheme to obtain fluxes along Y'''

        rhoFactorDiagonal = 0.0
        rhoVelFactorDiagonal = numpy.zeros(3)
        rhoEnergyFactorDiagonal = 0.0
        fpdiag = 0.0

        directionIdx = 1
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
            coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
            jdxPlus  = jdx + ndx
            jdxMinus = jdx + 1 - ndx  # Note the +1 for the faces
            rhoFactorDiagonal += coeff * \
                      ( self.rho.data[idx,jdxMinus,kdx] * \
                        self.velocity.data[idx,jdxMinus,kdx,directionIdx] + \
                        self.rho.data[idx,jdxPlus,kdx] * \
                        self.velocity.data[idx,jdxPlus,kdx,directionIdx] )
            rhoVelFactorDiagonal[:] += coeff * \
                      ( self.rhoVel.data[idx,jdxMinus,kdx,:] * \
                        self.velocity.data[idx,jdxMinus,kdx,directionIdx] + \
                        self.rhoVel.data[idx,jdxPlus,kdx,:] * \
                        self.velocity.data[idx,jdxPlus,kdx,directionIdx] )
            rhoEnergyFactorDiagonal += coeff * \
                      ( rhoEnthalpy.data[idx,jdxMinus,kdx] * \
                        self.velocity.data[idx,jdxMinus,kdx,directionIdx] + \
                        rhoEnthalpy.data[idx,jdxPlus,kdx] * \
                        self.velocity.data[idx,jdxPlus,kdx,directionIdx] )
            fpdiag += coeff * \
                      ( self.pressure.data[idx,jdxMinus,kdx] + \
                        self.pressure.data[idx,jdxPlus,kdx] )
        # Bilinear fluxes
        rhoFactorSkew = 0.0
        rhoVelFactorSkew = numpy.zeros(3)
        rhoEnergyFactorSkew = 0.0

        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeCoeffs)
        #  mdx = -N+1,...,0
        for mdx in range(2-numberCoeffs,1):
          tmp = 0.0
          jdxPlus = jdx + mdx
          for ndx in range(1,mdx+numberCoeffs):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[ndx-mdx] * \
                   self.velocity.data[idx,jdx+ndx,kdx,directionIdx]

          rhoFactorSkew += self.rho.data[idx,jdxPlus,kdx] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idx,jdxPlus,kdx,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idx,jdxPlus,kdx] * tmp
        #  mdx = 1,...,N
        for mdx in range(1,numberCoeffs):
          tmp = 0.0
          jdxPlus = jdx + mdx
          for ndx in range(mdx-numberCoeffs+1,1):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[mdx-ndx] * \
                   self.velocity.data[idx,jdx+ndx,kdx,directionIdx]

          rhoFactorSkew += self.rho.data[idx,jdxPlus,kdx] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idx,jdxPlus,kdx,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idx,jdxPlus,kdx] * tmp

        #  Final split fluxes
        splitParameter = self.mesh.spatialStencils.split
        oneMinusSplitParameter = 1.0 - splitParameter
        rhoFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoFactorDiagonal + \
          oneMinusSplitParameter * rhoFactorSkew
        rhoVelFlux.data[idx,jdx,kdx,:] = \
          splitParameter * rhoVelFactorDiagonal[:] + \
          oneMinusSplitParameter * rhoVelFactorSkew[:]
        rhoEnergyFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoEnergyFactorDiagonal + \
          oneMinusSplitParameter * rhoEnergyFactorSkew

        rhoVelFlux.data[idx,jdx,kdx,directionIdx] += fpdiag

    def InterpolateCentralZ(self, idx, jdx, kdx,
                            rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy):
        '''Interpolate using central scheme to obtain fluxes along Z'''

        rhoFactorDiagonal = 0.0
        rhoVelFactorDiagonal = numpy.zeros(3)
        rhoEnergyFactorDiagonal = 0.0
        fpdiag = 0.0

        directionIdx = 2
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
            coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
            kdxPlus  = kdx + ndx
            kdxMinus = kdx + 1 - ndx  # Note the +1 for the faces
            rhoFactorDiagonal += coeff * \
                      ( self.rho.data[idx,jdx,kdxMinus] * \
                        self.velocity.data[idx,jdx,kdxMinus,directionIdx] + \
                        self.rho.data[idx,jdx,kdxPlus] * \
                        self.velocity.data[idx,jdx,kdxPlus,directionIdx] )
            rhoVelFactorDiagonal[:] += coeff * \
                      ( self.rhoVel.data[idx,jdx,kdxMinus,:] * \
                        self.velocity.data[idx,jdx,kdxMinus,directionIdx] + \
                        self.rhoVel.data[idx,jdx,kdxPlus,:] * \
                        self.velocity.data[idx,jdx,kdxPlus,directionIdx] )
            rhoEnergyFactorDiagonal += coeff * \
                      ( rhoEnthalpy.data[idx,jdx,kdxMinus] * \
                        self.velocity.data[idx,jdx,kdxMinus,directionIdx] + \
                        rhoEnthalpy.data[idx,jdx,kdxPlus] * \
                        self.velocity.data[idx,jdx,kdxPlus,directionIdx] )
            fpdiag += coeff * \
                      ( self.pressure.data[idx,jdx,kdxMinus] + \
                        self.pressure.data[idx,jdx,kdxPlus] )
        # Bilinear fluxes
        rhoFactorSkew = 0.0
        rhoVelFactorSkew = numpy.zeros(3)
        rhoEnergyFactorSkew = 0.0

        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeCoeffs)
        #  mdx = -N+1,...,0
        for mdx in range(2-numberCoeffs,1):
          tmp = 0.0
          kdxPlus = kdx + mdx
          for ndx in range(1,mdx+numberCoeffs):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[ndx-mdx] * \
                   self.velocity.data[idx,jdx,kdx+ndx,directionIdx]

          rhoFactorSkew += self.rho.data[idx,jdx,kdxPlus] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idx,jdx,kdxPlus,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idx,jdx,kdxPlus] * tmp
        #  mdx = 1,...,N
        for mdx in range(1,numberCoeffs):
          tmp = 0.0
          kdxPlus = kdx + mdx
          for ndx in range(mdx-numberCoeffs+1,1):
            tmp += self.mesh.spatialStencils.firstDerivativeCoeffs[mdx-ndx] * \
                   self.velocity.data[idx,jdx,kdx+ndx,directionIdx]

          rhoFactorSkew += self.rho.data[idx,jdx,kdxPlus] * tmp
          rhoVelFactorSkew[:] += self.rhoVel.data[idx,jdx,kdxPlus,:] * tmp
          rhoEnergyFactorSkew += rhoEnthalpy.data[idx,jdx,kdxPlus] * tmp

        #  Final split fluxes
        splitParameter = self.mesh.spatialStencils.split
        oneMinusSplitParameter = 1.0 - splitParameter
        rhoFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoFactorDiagonal + \
          oneMinusSplitParameter * rhoFactorSkew
        rhoVelFlux.data[idx,jdx,kdx,:] = \
          splitParameter * rhoVelFactorDiagonal[:] + \
          oneMinusSplitParameter * rhoVelFactorSkew[:]
        rhoEnergyFlux.data[idx,jdx,kdx] = \
          splitParameter * rhoEnergyFactorDiagonal + \
          oneMinusSplitParameter * rhoEnergyFactorSkew

        rhoVelFlux.data[idx,jdx,kdx,directionIdx] += fpdiag

    def AddInviscid(self, rho_t, rhoVel_t, rhoEnergy_t):
        '''Add inviscid terms to time derivative of conserved flow fields'''

        # Warning: NEEDS UPDATED GHOST POINTS

        # Initialize temporary fields
        rhoFlux         = Field(self.mesh,1)
        rhoVelFlux      = Field(self.mesh,3)
        rhoEnergyFlux   = Field(self.mesh,1)
        rhoEnthalpy     = Field(self.mesh,1)

        # Compute total enthalpy at all points (including ghost)
        rhoEnthalpy.data[:,:,:] = self.rhoEnergy.data[:,:,:] + \
                                  self.pressure.data[:,:,:]

        # X direction
        # Interpolate using central scheme at all faces in X direction
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx-1,self.mesh.endXdx):
                    self.InterpolateCentralX(idx, jdx, kdx,
                      rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy)
        # Compute contribution to dQdt
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    rho_t.data[idx,jdx,kdx] -= \
                      ( rhoFlux.data[idx,jdx,kdx] - \
                        rhoFlux.data[idx-1,jdx,kdx] ) / \
                      self.mesh.dX[idx]
                    rhoVel_t.data[idx,jdx,kdx,:] -= \
                      ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                        rhoVelFlux.data[idx-1,jdx,kdx,:] ) / \
                      self.mesh.dX[idx]
                    rhoEnergy_t.data[idx,jdx,kdx] -= \
                      ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                        rhoEnergyFlux.data[idx-1,jdx,kdx] ) / \
                      self.mesh.dX[idx]
               
        # Y direction
        # Interpolate using central scheme at all faces in Y direction
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx-1,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    self.InterpolateCentralY(idx, jdx, kdx,
                      rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy)
        # Compute contribution to dQdt
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    rho_t.data[idx,jdx,kdx] -= \
                      ( rhoFlux.data[idx,jdx,kdx] - \
                        rhoFlux.data[idx,jdx-1,kdx] ) / \
                      self.mesh.dY[jdx]
                    rhoVel_t.data[idx,jdx,kdx,:] -= \
                      ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                        rhoVelFlux.data[idx,jdx-1,kdx,:] ) / \
                      self.mesh.dY[jdx]
                    rhoEnergy_t.data[idx,jdx,kdx] -= \
                      ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                        rhoEnergyFlux.data[idx,jdx-1,kdx] ) / \
                      self.mesh.dY[jdx]

        # Z direction
        # Interpolate using central scheme at all faces in Z direction
        for kdx in range(self.mesh.startZdx-1,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    self.InterpolateCentralZ(idx, jdx, kdx,
                      rhoFlux, rhoVelFlux, rhoEnergyFlux, rhoEnthalpy)
        # Compute contribution to dQdt
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    rho_t.data[idx,jdx,kdx] -= \
                      ( rhoFlux.data[idx,jdx,kdx] - \
                        rhoFlux.data[idx,jdx,kdx-1] ) / \
                      self.mesh.dZ[kdx]
                    rhoVel_t.data[idx,jdx,kdx,:] -= \
                      ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                        rhoVelFlux.data[idx,jdx,kdx-1,:] ) / \
                      self.mesh.dZ[kdx]
                    rhoEnergy_t.data[idx,jdx,kdx] -= \
                      ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                        rhoEnergyFlux.data[idx,jdx,kdx-1] ) / \
                      self.mesh.dZ[kdx]

        
    def ViscousFluxX(self, idx, jdx, kdx, rhoVelFlux, rhoEnergyFlux):
        '''Viscous flux in X coordinate direction'''

        # Interpolate dynamic viscosity to face (linear)
        muFace = 0.5 * (self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx,jdx,kdx]) + \
                        self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx+1,jdx,kdx]))
        # Interpolate velocity and its derivatives to face
        velocityFace    = numpy.zeros(3)
        velocityX_YFace = 0.0
        velocityX_ZFace = 0.0
        velocityY_YFace = 0.0
        velocityZ_ZFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
          idxPlus  = idx + ndx
          idxMinus = idx + 1 - ndx  # Note the +1 for the faces
          velocityFace += coeff * \
            ( self.velocity.data[idxMinus,jdx,kdx,:] + \
              self.velocity.data[idxPlus,jdx,kdx,:] )
          velocityX_YFace += coeff * \
            ( self.velocityGradientY.data[idxMinus,jdx,kdx,0] + \
              self.velocityGradientY.data[idxPlus,jdx,kdx,0] )
          velocityX_ZFace += coeff * \
            ( self.velocityGradientZ.data[idxMinus,jdx,kdx,0] + \
              self.velocityGradientZ.data[idxPlus,jdx,kdx,0] )
          velocityY_YFace += coeff * \
            ( self.velocityGradientY.data[idxMinus,jdx,kdx,1] + \
              self.velocityGradientY.data[idxPlus,jdx,kdx,1] )
          velocityZ_ZFace += coeff * \
            ( self.velocityGradientZ.data[idxMinus,jdx,kdx,2] + \
              self.velocityGradientZ.data[idxPlus,jdx,kdx,2] )

        # Differentiate at face
        velocityX_XFace = 0.0
        velocityY_XFace = 0.0
        velocityZ_XFace = 0.0
        temperature_XFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeFaceCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.firstDerivativeFaceCoeffs[ndx]
          idxPlus  = idx + ndx
          idxMinus = idx + 1 - ndx # Note the +1 for the faces
          velocityX_XFace += coeff * \
            ( self.velocity.data[idxPlus,jdx,kdx,0] - \
              self.velocity.data[idxMinus,jdx,kdx,0] )
          velocityY_XFace += coeff * \
            ( self.velocity.data[idxPlus,jdx,kdx,1] - \
              self.velocity.data[idxMinus,jdx,kdx,1] )
          velocityZ_XFace += coeff * \
            ( self.velocity.data[idxPlus,jdx,kdx,2] - \
              self.velocity.data[idxMinus,jdx,kdx,2] )
          temperature_XFace += coeff * \
            ( self.temperature.data[idxPlus,jdx,kdx] - \
              self.temperature.data[idxMinus,jdx,kdx] )
        dXFaceInverse = 1.0/self.mesh.dXFace[idx]
        velocityX_XFace   *= dXFaceInverse 
        velocityY_XFace   *= dXFaceInverse 
        velocityZ_XFace   *= dXFaceInverse 
        temperature_XFace *= dXFaceInverse 

        # Tensor components (at face)
        sigmaXX = muFace * ( 4.0 * velocityX_XFace - \
                             2.0 * velocityY_YFace - \
                             2.0 * velocityZ_ZFace ) / 3.0
        sigmaYX = muFace * ( velocityY_XFace + velocityX_YFace )
        sigmaZX = muFace * ( velocityZ_XFace + velocityX_ZFace )
        usigma = velocityFace[0] * sigmaXX + \
                 velocityFace[1] * sigmaYX + \
                 velocityFace[2] * sigmaZX
        heatFlux = - self.fluid.cpOverPrandtl * muFace * temperature_XFace

        # Fluxes
        rhoVelFlux.data[idx,jdx,kdx,0] = sigmaXX
        rhoVelFlux.data[idx,jdx,kdx,1] = sigmaYX
        rhoVelFlux.data[idx,jdx,kdx,2] = sigmaZX
        rhoEnergyFlux.data[idx,jdx,kdx] = usigma - heatFlux
        # WARNING: Add SGS terms

    def ViscousFluxY(self, idx, jdx, kdx, rhoVelFlux, rhoEnergyFlux):
        '''Viscous flux in Y coordinate direction'''

        # Interpolate dynamic viscosity to face (linear)
        muFace = 0.5 * (self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx,jdx,kdx]) + \
                        self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx,jdx+1,kdx]))

        # Interpolate velocity and its derivatives to face
        velocityFace    = numpy.zeros(3)
        velocityY_XFace = 0.0
        velocityY_ZFace = 0.0
        velocityX_XFace = 0.0
        velocityZ_ZFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
          jdxPlus  = jdx + ndx
          jdxMinus = jdx + 1 - ndx  # Note the +1 for the faces
          velocityFace += coeff * \
            ( self.velocity.data[idx,jdxMinus,kdx,:] + \
              self.velocity.data[idx,jdxPlus,kdx,:] )
          velocityY_XFace += coeff * \
            ( self.velocityGradientX.data[idx,jdxMinus,kdx,1] + \
              self.velocityGradientX.data[idx,jdxPlus,kdx,1] )
          velocityY_ZFace += coeff * \
            ( self.velocityGradientZ.data[idx,jdxMinus,kdx,1] + \
              self.velocityGradientZ.data[idx,jdxPlus,kdx,1] )
          velocityX_XFace += coeff * \
            ( self.velocityGradientX.data[idx,jdxMinus,kdx,0] + \
              self.velocityGradientX.data[idx,jdxPlus,kdx,0] )
          velocityZ_ZFace += coeff * \
            ( self.velocityGradientZ.data[idx,jdxMinus,kdx,2] + \
              self.velocityGradientZ.data[idx,jdxPlus,kdx,2] )

        # Differentiate at face
        velocityX_YFace = 0.0
        velocityY_YFace = 0.0
        velocityZ_YFace = 0.0
        temperature_YFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeFaceCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.firstDerivativeFaceCoeffs[ndx]
          jdxPlus  = jdx + ndx
          jdxMinus = jdx + 1 - ndx # Note the +1 for the faces
          velocityX_YFace += coeff * \
            ( self.velocity.data[idx,jdxPlus,kdx,0] - \
              self.velocity.data[idx,jdxMinus,kdx,0] )
          velocityY_YFace += coeff * \
            ( self.velocity.data[idx,jdxPlus,kdx,1] - \
              self.velocity.data[idx,jdxMinus,kdx,1] )
          velocityZ_YFace += coeff * \
            ( self.velocity.data[idx,jdxPlus,kdx,2] - \
              self.velocity.data[idx,jdxMinus,kdx,2] )
          temperature_YFace += coeff * \
            ( self.temperature.data[idx,jdxPlus,kdx] - \
              self.temperature.data[idx,jdxMinus,kdx] )
        dYFaceInverse = 1.0/self.mesh.dYFace[jdx]
        velocityX_YFace   *= dYFaceInverse 
        velocityY_YFace   *= dYFaceInverse 
        velocityZ_YFace   *= dYFaceInverse 
        temperature_YFace *= dYFaceInverse 

        # Tensor components (at face)
        sigmaXY = muFace * ( velocityX_YFace + velocityY_XFace )
        sigmaYY = muFace * ( 4.0 * velocityY_YFace - \
                             2.0 * velocityX_XFace - \
                             2.0 * velocityZ_ZFace ) / 3.0
        sigmaZY = muFace * ( velocityZ_YFace + velocityY_ZFace )
        usigma = velocityFace[0] * sigmaXY + \
                 velocityFace[1] * sigmaYY + \
                 velocityFace[2] * sigmaZY
        heatFlux = - self.fluid.cpOverPrandtl * muFace * temperature_YFace

        # Fluxes
        rhoVelFlux.data[idx,jdx,kdx,0] = sigmaXY
        rhoVelFlux.data[idx,jdx,kdx,1] = sigmaYY
        rhoVelFlux.data[idx,jdx,kdx,2] = sigmaZY
        rhoEnergyFlux.data[idx,jdx,kdx] = usigma - heatFlux

        # WARNING: Add SGS terms
        
    def ViscousFluxZ(self, idx, jdx, kdx, rhoVelFlux, rhoEnergyFlux):
        '''Viscous flux in Z coordinate direction'''

        # Interpolate dynamic viscosity to face (linear)
        muFace = 0.5 * (self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx,jdx,kdx]) + \
                        self.fluid.GetDynamicViscosity(
                          self.temperature.data[idx,jdx+1,kdx]))

        # Interpolate velocity and its derivatives to face
        velocityFace    = numpy.zeros(3)
        velocityZ_XFace = 0.0
        velocityZ_YFace = 0.0
        velocityX_XFace = 0.0
        velocityY_YFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.interpolateCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.interpolateCoeffs[ndx]
          kdxPlus  = kdx + ndx
          kdxMinus = kdx + 1 - ndx  # Note the +1 for the faces
          velocityFace += coeff * \
            ( self.velocity.data[idx,jdx,kdxMinus,:] + \
              self.velocity.data[idx,jdx,kdxPlus,:] )
          velocityZ_XFace += coeff * \
            ( self.velocityGradientX.data[idx,jdx,kdxMinus,2] + \
              self.velocityGradientX.data[idx,jdx,kdxPlus,2] )
          velocityZ_YFace += coeff * \
            ( self.velocityGradientY.data[idx,jdx,kdxMinus,2] + \
              self.velocityGradientY.data[idx,jdx,kdxPlus,2] )
          velocityX_XFace += coeff * \
            ( self.velocityGradientX.data[idx,jdx,kdxMinus,0] + \
              self.velocityGradientX.data[idx,jdx,kdxPlus,0] )
          velocityY_YFace += coeff * \
            ( self.velocityGradientY.data[idx,jdx,kdxMinus,1] + \
              self.velocityGradientY.data[idx,jdx,kdxPlus,1] )

        # Differentiate at face
        velocityX_ZFace = 0.0
        velocityY_ZFace = 0.0
        velocityZ_ZFace = 0.0
        temperature_ZFace = 0.0
        numberCoeffs = len(self.mesh.spatialStencils.firstDerivativeFaceCoeffs)
        for ndx in range(1, numberCoeffs):
          coeff = self.mesh.spatialStencils.firstDerivativeFaceCoeffs[ndx]
          kdxPlus  = kdx + ndx
          kdxMinus = kdx + 1 - ndx # Note the +1 for the faces
          velocityX_ZFace += coeff * \
            ( self.velocity.data[idx,jdx,kdxPlus,0] - \
              self.velocity.data[idx,jdx,kdxMinus,0] )
          velocityY_ZFace += coeff * \
            ( self.velocity.data[idx,jdx,kdxPlus,1] - \
              self.velocity.data[idx,jdx,kdxMinus,1] )
          velocityZ_ZFace += coeff * \
            ( self.velocity.data[idx,jdx,kdxPlus,2] - \
              self.velocity.data[idx,jdx,kdxMinus,2] )
          temperature_ZFace += coeff * \
            ( self.temperature.data[idx,jdx,kdxPlus] - \
              self.temperature.data[idx,jdx,kdxMinus] )
        dZFaceInverse = 1.0/self.mesh.dZFace[kdx]
        velocityX_ZFace   *= dZFaceInverse 
        velocityY_ZFace   *= dZFaceInverse 
        velocityZ_ZFace   *= dZFaceInverse 
        temperature_ZFace *= dZFaceInverse 

        # Tensor components (at face)
        sigmaXZ = muFace * ( velocityX_ZFace + velocityZ_XFace )
        sigmaYZ = muFace * ( velocityY_ZFace + velocityZ_YFace )
        sigmaZZ = muFace * ( 4.0 * velocityZ_ZFace - \
                             2.0 * velocityX_XFace - \
                             2.0 * velocityY_YFace ) / 3.0
        usigma = velocityFace[0] * sigmaXZ + \
                 velocityFace[1] * sigmaYZ + \
                 velocityFace[2] * sigmaZZ
        heatFlux = - self.fluid.cpOverPrandtl * muFace * temperature_ZFace

        # Fluxes
        rhoVelFlux.data[idx,jdx,kdx,0] = sigmaXZ
        rhoVelFlux.data[idx,jdx,kdx,1] = sigmaYZ
        rhoVelFlux.data[idx,jdx,kdx,2] = sigmaZZ
        rhoEnergyFlux.data[idx,jdx,kdx] = usigma - heatFlux

        # WARNING: Add SGS terms
        

    def AddViscous(self, rho_t, rhoVel_t, rhoEnergy_t):
        '''Add viscous terms to time derivative of conserved flow fields'''

        # Warning: NEEDS UPDATED GHOST POINTS

        # Initialize temporary fields
        rhoVelFlux      = Field(self.mesh,3)
        rhoEnergyFlux   = Field(self.mesh,1)
        rhoEnthalpy     = Field(self.mesh,1)

        # X direction
        # Compute viscous flux in X at faces
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx-1,self.mesh.endXdx):
                    self.ViscousFluxX(idx, jdx, kdx,
                      rhoVelFlux, rhoEnergyFlux)
        # Add X contribution to DqDt 
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                    dXInv = 1.0 / self.mesh.dX[idx]
                    rhoVel_t.data[idx,jdx,kdx,:] += \
                      dXInv * ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                                rhoVelFlux.data[idx-1,jdx,kdx,:] )
                    rhoEnergy_t.data[idx,jdx,kdx] += \
                      dXInv * ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                                rhoEnergyFlux.data[idx-1,jdx,kdx] )

        # Y direction
        # Compute viscous flux in Y at faces
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx-1,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    self.ViscousFluxY(idx, jdx, kdx,
                      rhoVelFlux, rhoEnergyFlux)
        # Add Y contribution to DqDt 
        for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
            dYInv = 1.0 / self.mesh.dY[jdx]
            for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    rhoVel_t.data[idx,jdx,kdx,:] += \
                      dYInv * ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                                rhoVelFlux.data[idx,jdx-1,kdx,:] )
                    rhoEnergy_t.data[idx,jdx,kdx] += \
                      dYInv * ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                                rhoEnergyFlux.data[idx,jdx-1,kdx] )

        # Z direction
        # Compute viscous flux in Z at faces
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx-1,self.mesh.endZdx):
                    self.ViscousFluxZ(idx, jdx, kdx,
                      rhoVelFlux, rhoEnergyFlux)
        # Add Z contribution to DqDt 
        for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
            dZInv = 1.0 / self.mesh.dZ[kdx]
            for idx in range(self.mesh.startXdx,self.mesh.endXdx):
                for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                    rhoVel_t.data[idx,jdx,kdx,:] += \
                      dZInv * ( rhoVelFlux.data[idx,jdx,kdx,:] - \
                                rhoVelFlux.data[idx,jdx,kdx-1,:] )
                    rhoEnergy_t.data[idx,jdx,kdx] += \
                      dZInv * ( rhoEnergyFlux.data[idx,jdx,kdx] - \
                                rhoEnergyFlux.data[idx-1,jdx,kdx-1] )

    def UpdateAuxiliaryVelocity(self):
        '''Update auxiliary variables related to velocity'''

        # Computes velocities, and sets conserved variables and velocities on
        # ghost points

        # Compute velocity from momenta and density
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    self.velocity.data[idx,jdx,kdx,:] = \
                      self.rhoVel.data[idx,jdx,kdx,:] / \
                      self.rho.data[idx,jdx,kdx]

        # Update ghost points prior to gradient computations (could be done in
        # parallel with the velocity computation above, once the points near the
        # internal/periodic boundaries have been updated)
        # Conserved variables
        self.rho.UpdateGhost()
        self.rhoVel.UpdateGhost()
        self.rhoEnergy.UpdateGhost()
        # Velocities
        self.velocity.UpdateGhost()

        # Compute velocity gradients
        self.velocity.GradientX(self.velocityGradientX)
        self.velocity.GradientY(self.velocityGradientY)
        self.velocity.GradientZ(self.velocityGradientZ)
        
    def UpdateAuxiliaryThermodynamics(self):
        '''Update auxiliary variables related to thermodynamics'''

        # Computes pressure and temperature, and sets them on ghost points
        # WARNING: Assumes velocities on ghost points are updated

        # Compute velocity from momenta and density
        for idx in range(self.mesh.startXdx,self.mesh.endXdx):
            for jdx in range(self.mesh.startYdx,self.mesh.endYdx):
                for kdx in range(self.mesh.startZdx,self.mesh.endZdx):
                    # Kinetic energy from velocity
                    kineticEnergy = 0.5 * self.rho.data[idx,jdx,kdx] * \
                      numpy.dot(self.velocity.data[idx,jdx,kdx,:],
                                self.velocity.data[idx,jdx,kdx,:])
                    # Pressure from internal energy 
                    # ( internal energy = total energy - kinetic energy )
                    # and gas constant
                    self.pressure.data[idx,jdx,kdx] = \
                      self.fluid.gammaMinus1 * \
                      ( self.rhoEnergy.data[idx,jdx,kdx] - \
                        kineticEnergy )
                    # WARNING Add SGS contribution to pressure before computing
                    #         temperature
                    # Temperature from equation of state
                    self.temperature.data[idx,jdx,kdx] = \
                      self.pressure.data[idx,jdx,kdx] / \
                      ( self.fluid.gasConstant * self.rho.data[idx,jdx,kdx] )

	# Update ghost points (could be done in parallel with the velocity
	# computation above, once the points near the internal/periodic
	# boundaries have been updated )
	self.pressure.UpdateGhost()
        self.temperature.UpdateGhost()

def InterpolateFields(fields,mesh,pointCoordinates,
                      interpolationType='trilinear'):
    '''Interpolate fields at a given point within their domain'''

    # WARNING: All fields are assumed to have the same underlying mesh
    
    # Check domain is within bounds
    if pointCoordinates[0] < \
       mesh.coorXWithGhost[mesh.startXdx] or \
       pointCoordinates[0] > \
       mesh.coorXWithGhost[mesh.endXdx] or \
       pointCoordinates[1] < \
       mesh.coorYWithGhost[mesh.startYdx] or \
       pointCoordinates[1] > \
       mesh.coorYWithGhost[mesh.endYdx] or \
       pointCoordinates[2] < \
       mesh.coorZWithGhost[mesh.startZdx] or \
       pointCoordinates[2] > \
       mesh.coorZWithGhost[mesh.endZdx]:
        raise RuntimeError('Point coordinates out of flow domain')

    # Interpolate
    if interpolationType == "nearest":
        # Find indices of nearest mesh point to requested point
        nearestXdx = \
          numpy.argmin(numpy.fabs(mesh.coorXWithGhost-\
                                  pointCoordinates[0]))
        nearestYdx = \
          numpy.argmin(numpy.fabs(mesh.coorYWithGhost-\
                                  pointCoordinates[1]))
        nearestZdx = \
          numpy.argmin(numpy.fabs(mesh.coorZWithGhost-\
                                  pointCoordinates[2]))
        # Get quantities at the retrieved indices
        try:
            # Init list with interpolated values
            # - Single elements for fields of dimension 1
            # - numpy array for fields of dimension > 1
            interpolatedData = []
            for field in fields:
                if field.dimensions == 1:
                    interpolatedField_XYZ = \
                      field.data[nearestXdx,nearestYdx,nearestZdx]
                else:
                    interpolatedField_XYZ = \
                      field.data[nearestXdx,nearestYdx,nearestZdx,:]

                # Append to output list
                interpolatedData.append(interpolatedField_XYZ)

        except:
            raise RuntimeError('Wrong indices')
    elif interpolationType == "trilinear":
        # Find indices of nearest (to the left = floor) 
        # mesh point to requested point
        floorXdx = numpy.argmin(\
          numpy.ma.masked_less(pointCoordinates[0]-mesh.coorXWithGhost,0.0))
        floorYdx = numpy.argmin(\
          numpy.ma.masked_less(pointCoordinates[1]-mesh.coorYWithGhost,0.0))
        floorZdx = numpy.argmin(\
          numpy.ma.masked_less(pointCoordinates[2]-mesh.coorZWithGhost,0.0))
        factorXdx = (pointCoordinates[0] - \
                     mesh.coorXWithGhost[floorXdx])/\
                    mesh.deltaX[floorXdx]
        factorYdx = (pointCoordinates[1] - \
                     mesh.coorYWithGhost[floorYdx])/\
                    mesh.deltaY[floorYdx]
        factorZdx = (pointCoordinates[2] - \
                     mesh.coorZWithGhost[floorZdx])/\
                    mesh.deltaZ[floorZdx]
        oneMinusFactorXdx = (1.0 - factorXdx)
        oneMinusFactorYdx = (1.0 - factorYdx)
        oneMinusFactorZdx = (1.0 - factorZdx)
        # Get velocity at the cell vertices
        try:
            # Init list with interpolated values
            # - Single elements for fields of dimension 1
            # - numpy array for fields of dimension > 1
            interpolatedData = []
            for field in fields:
                if field.dimensions == 1:
                    fieldAt000 = field.data[floorXdx,  floorYdx,  floorZdx  ]
                    fieldAt010 = field.data[floorXdx,  floorYdx+1,floorZdx  ]
                    fieldAt001 = field.data[floorXdx,  floorYdx,  floorZdx+1]
                    fieldAt011 = field.data[floorXdx,  floorYdx+1,floorZdx+1]
                    fieldAt100 = field.data[floorXdx+1,floorYdx,  floorZdx  ]
                    fieldAt110 = field.data[floorXdx+1,floorYdx+1,floorZdx  ]
                    fieldAt101 = field.data[floorXdx+1,floorYdx,  floorZdx+1]
                    fieldAt111 = field.data[floorXdx+1,floorYdx+1,floorZdx+1]
                else:
                    fieldAt000 = field.data[floorXdx,  floorYdx,  floorZdx  ,:]
                    fieldAt010 = field.data[floorXdx,  floorYdx+1,floorZdx  ,:]
                    fieldAt001 = field.data[floorXdx,  floorYdx,  floorZdx+1,:]
                    fieldAt011 = field.data[floorXdx,  floorYdx+1,floorZdx+1,:]
                    fieldAt100 = field.data[floorXdx+1,floorYdx,  floorZdx  ,:]
                    fieldAt110 = field.data[floorXdx+1,floorYdx+1,floorZdx  ,:]
                    fieldAt101 = field.data[floorXdx+1,floorYdx,  floorZdx+1,:]
                    fieldAt111 = field.data[floorXdx+1,floorYdx+1,floorZdx+1,:]
                # Interpolate along X
                interpolatedField_X00 = \
                  fieldAt000 * oneMinusFactorXdx + fieldAt100 * factorXdx
                interpolatedField_X10 = \
                  fieldAt010 * oneMinusFactorXdx + fieldAt110 * factorXdx
                interpolatedField_X01 = \
                  fieldAt001 * oneMinusFactorXdx + fieldAt101 * factorXdx
                interpolatedField_X11 = \
                  fieldAt011 * oneMinusFactorXdx + fieldAt111 * factorXdx
                # Interpolate along Y
                interpolatedField_XY0 = \
                  interpolatedField_X00 * oneMinusFactorYdx + \
                  interpolatedField_X10 * factorYdx
                interpolatedField_XY1 = \
                  interpolatedField_X01 * oneMinusFactorYdx + \
                  interpolatedField_X11 * factorYdx
                # Interpolate along Z
                interpolatedField_XYZ = \
                  interpolatedField_XY0 * oneMinusFactorZdx + \
                  interpolatedField_XY1 * factorZdx
                # Append to output list
                interpolatedData.append(interpolatedField_XYZ)

        except:
            raise RuntimeError('Wrong indices')
    else:
        raise RuntimeError('Interpolation type not implemented')
    return interpolatedData

# Particles

def Init1DArray(array, number):

    if type(array) == float:
        arrayOut = numpy.ones(number)*array
    elif type(array) == numpy.ndarray:
        if len(array) != number:
            raise RuntimeError('Lenght of array (%d) '
                               ' differs from number of particles (%d)', 
                               len(array), number)
        else:
            arrayOut = array
    elif type(array) == str:
        if array.split(':')[0:2] == ['random', 'uniform']:
            minX = float(array.split(':')[2])
            maxX = float(array.split(':')[3])
            arrayOut = numpy.random.uniform(minX,maxX,number)
        elif array.split(':')[0:2] == ['line', 'uniform']:
            minX = float(array.split(':')[2])
            maxX = float(array.split(':')[3])
            arrayOut = numpy.linspace(minX,maxX,number)
        else:
            raise RuntimeError('Type of array (%s) unknown',array)
    else:
        raise RuntimeError('Type of array (%s) unknown', array)

    return arrayOut

def Init3DArray(arrayXIn, arrayYIn, arrayZIn, number):

    arrayX = Init1DArray(arrayXIn, number)
    arrayY = Init1DArray(arrayYIn, number)
    arrayZ = Init1DArray(arrayZIn, number)

    arrayOut = numpy.empty((number,3))
    arrayOut[:,0] = arrayX[:]
    arrayOut[:,1] = arrayY[:]
    arrayOut[:,2] = arrayZ[:]

    return arrayOut

class ParticleDistribution:
    '''ParticleDistribution'''

    def __init__(self, number, type,
                 coorX, coorY, coorZ,
                 velocityX, velocityY, velocityZ,
                 diameter, density):

        log.info('Initializing particle distribution')

        self.number = number

        self.particlesType = type
         
        self.position = Init3DArray(coorX,coorY,coorZ,number)
        self.velocity = Init3DArray(velocityX,velocityY,velocityZ,number)

        self.diameter = Init1DArray(diameter,number)
        self.density = Init1DArray(density,number)

    def SetVelocitiesToFlow(self, flow):

        # Calculate fluid flow velocity at particle positions
        for particleIdx in range(self.number):
            # Interpolate flow quantities at given point
            [flowVelocity] = \
              InterpolateFields(\
                [flow.velocity],flow.mesh,
                self.position[particleIdx,:],interpolationType='trilinear')
            # Update time derivative of position for this particle
            # Kinematics
            self.velocity[particleIdx,:] = flowVelocity

    def AddFlowCoupling(self, position_t, velocity_t, flow):

        # Calculate fluid flow velocity at particle positions
        for particleIdx in range(self.number):
            # Interpolate flow quantities at given point
            [flowDensity, flowVelocity, flowTemperature] = \
              InterpolateFields(\
                [flow.rho, flow.velocity, flow.temperature],flow.mesh,
                self.position[particleIdx,:],interpolationType='trilinear')
            # Get flow dynamic viscosity from temperature
            flowDynamicViscosity = \
              flow.fluid.GetDynamicViscosity(flowTemperature)
            # Update time derivative of position for this particle
            if self.particlesType == 'flowTracer':
                # Flow tracer
                # - Kinematics
                position_t[particleIdx,:] += flowVelocity
                # - Momentum equation (no inertia for flow tracer)
                velocity_t[particleIdx,:] += 0.0
            elif self.particlesType == 'smallParticles':
                # Solid particles
                # - Kinematics
                position_t[particleIdx,:] += self.velocity[particleIdx,:]
                # - Momentum equation (Portela and Oliemans 2003)
                velocity_t[particleIdx,:] += \
                  18.0 * numpy.pi * flowDynamicViscosity / \
                  (self.density[particleIdx] * \
                   self.diameter[particleIdx]**2) * \
                  (flowVelocity - self.velocity[particleIdx,:])
            else:
                raise RuntimeError('Unknown particles type %s',
                                   self.particlesType)

    def UpdateAuxiliary(self,flow):

        # Apply periodicity if needed
        if flow.mesh.coorXType == "periodic":
            for particleIdx in range(self.number):
                if self.position[particleIdx,0] < \
                   flow.mesh.coorXWithGhost[flow.mesh.startXdx]:
                    self.position[particleIdx,0] += \
                      flow.mesh.coorXWithGhost[flow.mesh.endXdx]
                elif self.position[particleIdx,0] > \
                   flow.mesh.coorXWithGhost[flow.mesh.endXdx]:
                    self.position[particleIdx,0] -= \
                      flow.mesh.coorXWithGhost[flow.mesh.endXdx]
        if flow.mesh.coorYType == "periodic":
            for particleIdx in range(self.number):
                if self.position[particleIdx,1] < \
                   flow.mesh.coorYWithGhost[flow.mesh.startYdx]:
                    self.position[particleIdx,1] += \
                      flow.mesh.coorYWithGhost[flow.mesh.endYdx]
                elif self.position[particleIdx,1] > \
                   flow.mesh.coorYWithGhost[flow.mesh.endYdx]:
                    self.position[particleIdx,1] -= \
                      flow.mesh.coorYWithGhost[flow.mesh.endYdx]
        if flow.mesh.coorZType == "periodic":
            for particleIdx in range(self.number):
                if self.position[particleIdx,2] < \
                   flow.mesh.coorZWithGhost[flow.mesh.startZdx]:
                    self.position[particleIdx,2] += \
                      flow.mesh.coorZWithGhost[flow.mesh.endZdx]
                elif self.position[particleIdx,2] > \
                   flow.mesh.coorZWithGhost[flow.mesh.endZdx]:
                    self.position[particleIdx,2] -= \
                      flow.mesh.coorZWithGhost[flow.mesh.endZdx]

    def WriteOutput(self,outputFileName):
        # Write distribution of particles (one line per particle)
        with open(outputFileName, 'w') as outFile:
            # Write header
            outFile.write("# ID diameter density "
                          "coorX coorY coorZ "
                          "velocityX velocityY velocityZ "
                          "velocityX velocityY velocityZ ")
            for ndx in range(self.number):
                outFile.write(str(ndx) + " " + \
                              str(self.diameter[ndx]) + " " + \
                              str(self.density[ndx]) + " " + \
                              str(self.position[ndx,0]) + " " + \
                              str(self.position[ndx,1]) + " " + \
                              str(self.position[ndx,2]) + " " + \
                              str(self.velocity[ndx,0]) + " " + \
                              str(self.velocity[ndx,1]) + " " + \
                              str(self.velocity[ndx,2]) + "\n")
            outFile.write("\n")

class Mesh:
    '''Mesh'''

    def __init__(self, inputFileName, spatialStencils):

        log.info('Initializing mesh')

        self.spatialStencils = spatialStencils

        # Read options from file
        dom = xml.dom.minidom.parse(inputFileName)
        meshElement = dom.getElementsByTagName("mesh")[0]
        self.meshType = meshElement.attributes["type"].value

        if self.meshType == "rectilinearGrid":
            # X
            coorXElement = meshElement.getElementsByTagName("coorX")[0]
            self.coorXSpacing = coorXElement.attributes["spacing"].value
            if self.coorXSpacing == "uniform":
                numPointsX = int(coorXElement.attributes["numPoints"].value)
                minX = float(coorXElement.attributes["min"].value)
                maxX = float(coorXElement.attributes["max"].value)
                self.coorXType = coorXElement.attributes["type"].value
                if self.coorXType == "periodic":
                    self.numGhostX = spatialStencils.order/2
                    self.coorX = numpy.linspace(minX,maxX,numPointsX+1)[:-1]
                    if numPointsX > 1:
                        interval = self.coorX[1]-self.coorX[0]
                        self.leftBoundaryX  = "periodic"
                        self.rightBoundaryX = "periodic"
                    else:
                        interval = maxX - minX
                        self.leftBoundaryX  = "symmetry"
                        self.rightBoundaryX = "symmetry"
                    self.coorXWithGhost = \
                      numpy.linspace(minX-self.numGhostX*interval,\
                                     maxX+self.numGhostX*interval,\
                                     numPointsX+2*self.numGhostX+1)[:-1]
                else:
                    raise RuntimeError('Unknown coorXType %s', self.coorXType)
            # Y
            coorYElement = meshElement.getElementsByTagName("coorY")[0]
            self.coorYSpacing = coorYElement.attributes["spacing"].value
            if self.coorYSpacing == "uniform":
                numPointsY = int(coorYElement.attributes["numPoints"].value)
                minY = float(coorYElement.attributes["min"].value)
                maxY = float(coorYElement.attributes["max"].value)
                self.coorYType = coorYElement.attributes["type"].value
                if self.coorYType == "periodic":
                    self.numGhostY = spatialStencils.order/2
                    self.coorY = numpy.linspace(minY,maxY,numPointsY+1)[:-1]
                    if numPointsY > 1:
                        interval = self.coorY[1]-self.coorY[0]
                        self.leftBoundaryY  = "periodic"
                        self.rightBoundaryY = "periodic"
                    else:
                        interval = maxY - minY
                        self.leftBoundaryY  = "symmetry"
                        self.rightBoundaryY = "symmetry"
                    self.coorYWithGhost = \
                      numpy.linspace(minY-self.numGhostY*interval,\
                                     maxY+self.numGhostY*interval,\
                                     numPointsY+2*self.numGhostY+1)[:-1]
                else:
                    raise RuntimeError('Unknown coorYType %s', self.coorYType)
            # Z
            coorZElement = meshElement.getElementsByTagName("coorZ")[0]
            self.coorZSpacing = coorZElement.attributes["spacing"].value
            if self.coorZSpacing == "uniform":
                numPointsZ = int(coorZElement.attributes["numPoints"].value)
                minZ = float(coorZElement.attributes["min"].value)
                maxZ = float(coorZElement.attributes["max"].value)
                self.coorZType = coorZElement.attributes["type"].value
                if self.coorZType == "periodic":
                    self.numGhostZ = spatialStencils.order/2
                    self.coorZ = numpy.linspace(minZ,maxZ,numPointsZ+1)[:-1]
                    if numPointsZ > 1:
                        interval = self.coorZ[1]-self.coorZ[0]
                        self.leftBoundaryZ  = "periodic"
                        self.rightBoundaryZ = "periodic"
                    else:
                        interval = maxZ - minZ
                        self.leftBoundaryZ  = "symmetry"
                        self.rightBoundaryZ = "symmetry"
                    self.coorZWithGhost = \
                      numpy.linspace(minZ-self.numGhostZ*interval,\
                                     maxZ+self.numGhostZ*interval,\
                                     numPointsZ+2*self.numGhostZ+1)[:-1]
                else:
                    raise RuntimeError('Unknown coorZType %s', self.coorZType)

        # Define points
        self.numPointsX = len(self.coorX)
        self.numPointsY = len(self.coorY)
        self.numPointsZ = len(self.coorZ)
        self.numPoints = self.numPointsX * self.numPointsY * self.numPointsZ

        # Define points with ghost
        self.numPointsXWithGhost = len(self.coorXWithGhost)
        self.numPointsYWithGhost = len(self.coorYWithGhost)
        self.numPointsZWithGhost = len(self.coorZWithGhost)
        self.numPointsWithGhost = self.numPointsXWithGhost * \
                                  self.numPointsYWithGhost * \
                                  self.numPointsZWithGhost

        # Define start and end indices
        self.startXdx = self.numGhostX
        self.endXdx = self.numPointsXWithGhost - self.numGhostX
        self.startYdx = self.numGhostY
        self.endYdx = self.numPointsYWithGhost - self.numGhostY
        self.startZdx = self.numGhostZ
        self.endZdx = self.numPointsZWithGhost - self.numGhostZ

        # Define grid metrics
        # Note: - Ghost elements should never be used; they are only defined
        #         for index consistency
        self.deltaX = self.coorXWithGhost[1:] - self.coorXWithGhost[:-1]
        self.deltaY = self.coorYWithGhost[1:] - self.coorYWithGhost[:-1]
        self.deltaZ = self.coorZWithGhost[1:] - self.coorZWithGhost[:-1]
        self.dX = numpy.zeros(self.numPointsXWithGhost)
        self.dY = numpy.zeros(self.numPointsYWithGhost)
        self.dZ = numpy.zeros(self.numPointsZWithGhost)
#        self.d2X = numpy.zeros(self.numPointsXWithGhost)
#        self.d2Y = numpy.zeros(self.numPointsYWithGhost)
#        self.d2Z = numpy.zeros(self.numPointsZWithGhost)
        self.dXFace = numpy.zeros(self.numPointsXWithGhost+1)
        self.dYFace = numpy.zeros(self.numPointsYWithGhost+1)
        self.dZFace = numpy.zeros(self.numPointsZWithGhost+1)
                      
        self.dX[self.startXdx:self.endXdx] = \
          numpy.convolve(self.coorXWithGhost,
                         self.spatialStencils.firstDerivativeConvolveCoeffs,
                         mode='valid')
        self.dY[self.startYdx:self.endYdx] = \
          numpy.convolve(self.coorYWithGhost,
                         self.spatialStencils.firstDerivativeConvolveCoeffs,
                         mode='valid')
        self.dZ[self.startZdx:self.endZdx] = \
          numpy.convolve(self.coorZWithGhost,
                         self.spatialStencils.firstDerivativeConvolveCoeffs,
                         mode='valid')

#        self.d2X[self.startXdx:self.endXdx] = \
#          numpy.convolve(self.coorXWithGhost,
#                         self.spatialStencils.secondDerivativeConvolveCoeffs,
#                         mode='valid')
#        self.d2Y[self.startYdx:self.endYdx] = \
#          numpy.convolve(self.coorYWithGhost,
#                         self.spatialStencils.firstDerivativeConvolveCoeffs,
#                         mode='valid')
#        self.d2Z[self.startZdx:self.endZdx] = \
#          numpy.convolve(self.coorZWithGhost,
#                         self.spatialStencils.secondDerivativeConvolveCoeffs,
#                         mode='valid')
#        self.dXFace[self.startXdx-1:self.endXdx] = \
#          numpy.convolve(self.coorXWithGhost,
#                         self.spatialStencils.firstDerivativeFaceConvolveCoeffs,
#                         mode='valid')
#        self.dYFace[self.startYdx-1:self.endYdx] = \
#          numpy.convolve(self.coorYWithGhost,
#                         self.spatialStencils.firstDerivativeFaceConvolveCoeffs,
#                         mode='valid')
#        self.dZFace[self.startZdx-1:self.endZdx] = \
#          numpy.convolve(self.coorZWithGhost,
#                         self.spatialStencils.firstDerivativeFaceConvolveCoeffs,
#                         mode='valid')


        numberCoeffs = len(self.spatialStencils.firstDerivativeFaceCoeffs)
        for idx in range(self.startXdx-1,self.endXdx):
            self.dXFace[idx] = 0.0
            for ndx in range(1, numberCoeffs):
                coeff = self.spatialStencils.firstDerivativeFaceCoeffs[ndx]
                self.dXFace[idx] += coeff * \
                          ( self.coorXWithGhost[idx+ndx] - \
                            self.coorXWithGhost[idx-ndx+1] )
        for jdx in range(self.startYdx-1,self.endYdx):
            self.dYFace[jdx] = 0.0
            for ndx in range(1, numberCoeffs):
                coeff = self.spatialStencils.firstDerivativeFaceCoeffs[ndx]
                self.dYFace[jdx] += coeff * \
                          ( self.coorYWithGhost[jdx+ndx] - \
                            self.coorYWithGhost[jdx-ndx+1] )
        for kdx in range(self.startZdx-1,self.endZdx):
            self.dZFace[kdx] = 0.0
            for ndx in range(1, numberCoeffs):
                coeff = self.spatialStencils.firstDerivativeFaceCoeffs[ndx]
                self.dZFace[kdx] += coeff * \
                          ( self.coorZWithGhost[kdx+ndx] - \
                            self.coorZWithGhost[kdx-ndx+1] )

class TimeIntegrator:
    '''TimeIntegrator'''

    def __init__(self,
                 inputFileName,
                 flow,
                 particles):

        # Read options from inputFile
        dom = xml.dom.minidom.parse(inputFileName)
        timeIntegratorElement = dom.getElementsByTagName("timeIntegrator")[0]
        self.type  = \
          timeIntegratorElement.getElementsByTagName("type")[0].firstChild.data
        self.maxTimeStep = \
          int(timeIntegratorElement.getElementsByTagName("maxTimeStep")[0].\
                firstChild.data)
        self.finalTime = \
          float(timeIntegratorElement.getElementsByTagName("finalTime")[0].\
                firstChild.data)
        self.cfl = \
          float(timeIntegratorElement.getElementsByTagName("cfl")[0].\
                firstChild.data)

        # Set reference to previously initialized flow object
        self.flow = flow
        # Set reference to previously initialized particle distribution object
        self.particles = particles
        # Reset time, timeStep, deltaTime and stage
        self.simulationTime = 0.0
        self.timeStep = 0
        self.deltaTime = -1
        self.stage = -1

        # Set coefficients for time stepping stages depending on type
        if self.type == "RungeKutta4":
            self.coeffTime = numpy.array([0.5, 0.5, 1.0, 1.0])
            self.coeffFunction = \
              numpy.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
        else:
            raise RuntimeError('Unknown type of time integrator %s', self.type)

    def CalculateDeltaTime(self):
        # Init spectral radii to zero
        convectiveSpectralRadius = 0.0
        viscousSpectralRadius = 0.0
        heatConductionSpectralRadius = 0.0
        mesh = self.flow.mesh
        for idx in range(mesh.startXdx,mesh.endXdx):
            for jdx in range(mesh.startYdx,mesh.endYdx):
                for kdx in range(mesh.startZdx,mesh.endZdx):
                    # Calculate equivalent cell diagonal from grid metrics
                    dXInverse = 1.0/mesh.dX[idx]
                    dYInverse = 1.0/mesh.dY[jdx]
                    dZInverse = 1.0/mesh.dZ[kdx]
                    dXYZInverseSquare = dXInverse*dXInverse + \
                                        dYInverse*dYInverse + \
                                        dZInverse*dZInverse
                    dXYZInverse = numpy.sqrt(dXYZInverseSquare)
                    # Convective spectral radii
                    convectiveSpectralRadius = numpy.max( \
                      [numpy.fabs(self.flow.velocity.data[idx,jdx,kdx,0])* \
                       dXInverse + \
                       numpy.fabs(self.flow.velocity.data[idx,jdx,kdx,1])* \
                       dYInverse + \
                       numpy.fabs(self.flow.velocity.data[idx,jdx,kdx,2])* \
                       dZInverse + \
                       self.flow.fluid.GetSoundSpeed(
                         self.flow.temperature.data[idx,jdx,kdx])*dXYZInverse,
                       convectiveSpectralRadius])

                    # Viscous spectral radii (including sgs model component)
                    dynamicViscosity = \
                      self.flow.fluid.GetDynamicViscosity(
                        self.flow.temperature.data[idx,jdx,kdx])
                    eddyViscosity = self.flow.sgsEddyViscosity.data[idx,jdx,kdx]
                    viscousSpectralRadius = numpy.max( \
                      [2.0 * ( dynamicViscosity + eddyViscosity ) / \
                       self.flow.rho.data[idx,jdx,kdx] * dXYZInverseSquare,
                       viscousSpectralRadius])
                    
                    # Heat conduction spectral radii (including sgs model 
                    # component)
                    kappa = self.flow.fluid.cp * dynamicViscosity / \
                            self.flow.fluid.prandtl
                    eddyKappa = self.flow.sgsEddyKappa.data[idx,jdx,kdx]
                    heatConductionSpectralRadius = numpy.max( \
                      [(kappa + eddyKappa) / \
                       (self.flow.fluid.cv * \
                        self.flow.rho.data[idx,jdx,kdx]) * dXYZInverseSquare,
                       heatConductionSpectralRadius])

        convectiveSpectralRadius *= \
          mesh.spatialStencils.firstDerivativeModifiedWaveNumber
        viscousSpectralRadius *= \
          mesh.spatialStencils.secondDerivativeModifiedWaveNumber
        heatConductionSpectralRadius *= \
          mesh.spatialStencils.secondDerivativeModifiedWaveNumber
        diffusiveSpectralRadius = numpy.max( \
          [viscousSpectralRadius, heatConductionSpectralRadius])
        spectralRadius = numpy.max(\
          [convectiveSpectralRadius, diffusiveSpectralRadius])

        self.deltaTime = self.cfl / spectralRadius

    def ComputeDFunctionDt(self, dFunctionDtArray):
        # Compute derivative of function array
        # Flow
        #   dFunctionDtArray[0] - rho_t
        #   dFunctionDtArray[1] - rhoVel_t
        #   dFunctionDtArray[2] - rhoEnergy_t
        # Particles
        #   dFunctionDtArray[3] - particlesPosition_t
        #   dFunctionDtArray[4] - particlesVelocity_t

        #log.info('Computing DFunctionDt in time stepper')
        # Reset to zero
        dFunctionDtArray[0].SetToConstant(0.0)
        dFunctionDtArray[1].SetToConstant(0.0)
        dFunctionDtArray[2].SetToConstant(0.0)
        dFunctionDtArray[3].fill(0.0)
        dFunctionDtArray[4].fill(0.0)
            
        ## Flow field
        # Add inviscid terms
        #log.info('Adding inviscid terms in flow solver')
        self.flow.AddInviscid( \
          dFunctionDtArray[0], \
          dFunctionDtArray[1], \
          dFunctionDtArray[2])

        # Update velocity gradients on ghost points
        self.flow.UpdateGhostVelocityGradient()
        # Add viscous terms
        #log.info('Adding viscous terms in flow solver')
        self.flow.AddViscous(  \
          dFunctionDtArray[0], \
          dFunctionDtArray[1], \
          dFunctionDtArray[2])

        # Add particle coupling on fluid flow
        # self.flow.AddParticleCoupling(
        #  dFunctionDtArray[0], \
        #  dFunctionDtArray[1], \
        #  dFunctionDtArray[2],
        #  self.particles)

        # Add radiation
        # self.flow.AddRadiation(
        #  dFunctionDtArray[0], \
        #  dFunctionDtArray[1], \
        #  dFunctionDtArray[2],
        #  self.particles)

        # Add body forces (e.g. gravity)
        # self.flow.AddBodyForces(
        #  dFunctionDtArray[0], \
        #  dFunctionDtArray[1], \
        #  dFunctionDtArray[2],
        #  self.particles)

        # Add flow coupling to particles
        self.particles.AddFlowCoupling(
          dFunctionDtArray[3],
          dFunctionDtArray[4],
          self.flow)

        ## Add body forces to particles
        #self.particles.AddBodyForces(
        # dFunctionDtArray[3], \
        # self.flow)

        # etc.
        
    def UpdateArrays(self, oldArray, newArray, dFunctionDtArray, solutionArray):
        if self.type == "RungeKutta4":
            if self.stage < 3:
                # Update arrays for this stage of time integrator
                for fdx in range(len(newArray)):
                    if dFunctionDtArray[fdx].__class__.__name__ == "Field":
                        newArray[fdx].data += self.coeffFunction[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx].data
                        solutionArray[fdx].CopyDataFrom(oldArray[fdx])
                        solutionArray[fdx].data += self.coeffTime[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx].data
                    elif dFunctionDtArray[fdx].__class__.__name__ == "ndarray":
                        newArray[fdx] += self.coeffFunction[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx]
                        solutionArray[fdx].fill(0.0)
                        solutionArray[fdx] += oldArray[fdx]
                        solutionArray[fdx] += self.coeffTime[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx]
            elif self.stage == 3:
                # Update arrays for this stage of time integrator
                for fdx in range(len(newArray)):
                    if dFunctionDtArray[fdx].__class__.__name__ == "Field":
                        solutionArray[fdx].CopyDataFrom(newArray[fdx])
                        solutionArray[fdx].data += self.coeffFunction[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx].data
                    elif dFunctionDtArray[fdx].__class__.__name__ == "ndarray":
                        solutionArray[fdx].fill(0.0)
                        solutionArray[fdx] += newArray[fdx]
                        solutionArray[fdx] += self.coeffFunction[self.stage] * \
                          self.deltaTime * dFunctionDtArray[fdx]

    def UpdateAuxiliary(self):
        # Update auxiliary variables after time integration step required for
        # the next time step computation

        # Compute auxiliary variables 
        # a) flow velocities and gradients
        self.flow.UpdateAuxiliaryVelocity()
        # WARNING: Add SGS contribution before updating pressure/temperature
        # b) flow velocities and gradients
        self.flow.UpdateAuxiliaryThermodynamics()

        # Update particles before next timestep
        self.particles.UpdateAuxiliary(self.flow)

    def AdvanceTimeStep(self):

        # Set-up additional fields
        # Old
        rho_old       = self.flow.rho.Copy() 
        rhoVel_old    = self.flow.rhoVel.Copy() 
        rhoEnergy_old = self.flow.rhoEnergy.Copy()
        particlesPosition_old = self.particles.position.copy()
        particlesVelocity_old = self.particles.velocity.copy()
        # New
        rho_new       = self.flow.rho.Copy() 
        rhoVel_new    = self.flow.rhoVel.Copy() 
        rhoEnergy_new = self.flow.rhoEnergy.Copy()
        particlesPosition_new = self.particles.position.copy()
        particlesVelocity_new = self.particles.velocity.copy()
        # Time derivative
        rho_t         = Field(self.flow.mesh,1)
        rhoVel_t      = Field(self.flow.mesh,3)
        rhoEnergy_t   = Field(self.flow.mesh,1)
        particlesPosition_t = numpy.empty((self.particles.number,3))
        particlesVelocity_t = numpy.empty((self.particles.number,3))

        # Pack into arrays
        oldArray         = [rho_old, rhoVel_old, rhoEnergy_old,
                             particlesPosition_old, particlesVelocity_old]
        newArray         = [rho_new, rhoVel_new, rhoEnergy_new,
                            particlesPosition_new, particlesVelocity_new]
        dFunctionDtArray = [rho_t, rhoVel_t, rhoEnergy_t,
                            particlesPosition_t, particlesVelocity_t]
        solutionArray    = \
          [self.flow.rho, self.flow.rhoVel, self.flow.rhoEnergy,
           self.particles.position, self.particles.velocity]
 
        # Apply time-stepping
        timeOld = self.simulationTime
        for self.stage in range(len(self.coeffFunction)):
            # Compute function derivative
            self.ComputeDFunctionDt(dFunctionDtArray)
            # Update arrays
            self.UpdateArrays(
              oldArray, newArray, dFunctionDtArray, solutionArray)
            self.simulationTime = \
              timeOld + self.coeffTime[self.stage] * self.deltaTime
            self.UpdateAuxiliary()

class SolverOptions:
    '''Solver options initialized through XML input file'''

    def __init__(self, inputFileName):

        #log.info('Reading solver options from %s', inputFileName)

        dom = xml.dom.minidom.parse(inputFileName)

        # Stencils
        spatialStencilsElement = dom.getElementsByTagName("spatialStencils")[0]
        self.spatialStencilsType = \
          spatialStencilsElement.getElementsByTagName("type")[0].firstChild.data
        self.spatialStencilsOrder = \
          int(spatialStencilsElement.getElementsByTagName("order")[0].\
                firstChild.data)
        self.spatialStencilsSplit = \
          float(spatialStencilsElement.getElementsByTagName("split")[0].\
                  firstChild.data)

        # Fluid
        fluidElement = dom.getElementsByTagName("fluid")[0]
        self.fluidType = fluidElement.attributes["type"].value
        if self.fluidType  == "gas": 
            self.fluidConstant = \
              float(fluidElement.getElementsByTagName("gasConstant")[0].\
                    firstChild.data)
            self.fluidGamma = \
              float(fluidElement.getElementsByTagName("gasGamma")[0].\
                    firstChild.data)
            self.fluidDynamicViscosityRef = \
              float(fluidElement.getElementsByTagName("dynamicViscosityRef")[0].\
                    firstChild.data)
            self.fluidDynamicViscosityTemperatureRef = \
              float(fluidElement.getElementsByTagName("dynamicViscosityTemperatureRef")[0].\
                    firstChild.data)
            self.fluidPrandtl= \
              float(fluidElement.getElementsByTagName("prandtl")[0].\
                    firstChild.data)
    
        # Particles
        # ---------
        particlesElement = dom.getElementsByTagName("particles")[0]
        # Number of particles
        self.particlesNumber = \
          particlesElement.getElementsByTagName("number")[0].firstChild.data
        # Type of particles
        self.particlesType = \
          particlesElement.getElementsByTagName("type")[0].firstChild.data
        # Coordinates
        particlesCoorXType = \
          particlesElement.getElementsByTagName("coorX")[0].\
            attributes["type"].value
        if particlesCoorXType == "constant":
            self.particlesCoorX = \
              float(particlesElement.getElementsByTagName("coorX")[0].\
                    firstChild.data)
        else:
            self.particlesCoorX = str(particlesCoorXType)
        particlesCoorYType = \
          particlesElement.getElementsByTagName("coorY")[0].\
            attributes["type"].value
        if particlesCoorYType == "constant":
            self.particlesCoorY = \
              float(particlesElement.getElementsByTagName("coorY")[0].\
                    firstChild.data)
        else:
            self.particlesCoorY = str(particlesCoorYType)
        particlesCoorZType = \
          particlesElement.getElementsByTagName("coorZ")[0].\
            attributes["type"].value
        if particlesCoorZType == "constant":
            self.particlesCoorZ = \
              float(particlesElement.getElementsByTagName("coorZ")[0].\
                    firstChild.data)
        else:
            self.particlesCoorZ = str(particlesCoorZType)
        # Velocities
        particlesVelocityXType = \
          particlesElement.getElementsByTagName("velocityX")[0].\
            attributes["type"].value
        if particlesVelocityXType == "constant":
            self.particlesVelocityX = \
              float(particlesElement.getElementsByTagName("velocityX")[0].\
                    firstChild.data)
        else:
            self.particlesVelocityX = str(particlesVelocityXType)
        particlesVelocityYType = \
          particlesElement.getElementsByTagName("velocityY")[0].\
            attributes["type"].value
        if particlesVelocityYType == "constant":
            self.particlesVelocityY = \
              float(particlesElement.getElementsByTagName("velocityY")[0].\
                    firstChild.data)
        else:
            self.particlesVelocityY = str(particlesVelocityYType)
        particlesVelocityZType = \
          particlesElement.getElementsByTagName("velocityZ")[0].\
            attributes["type"].value
        if particlesVelocityZType == "constant":
            self.particlesVelocityZ = \
              float(particlesElement.getElementsByTagName("velocityZ")[0].\
                    firstChild.data)
        else:
            self.particlesVelocityZ = str(particlesVelocityZType)
        # Diameter
        particlesDiameterType = \
          particlesElement.getElementsByTagName("diameter")[0].\
            attributes["type"].value
        if particlesDiameterType == "constant":
            self.particlesDiameter = \
              float(particlesElement.getElementsByTagName("diameter")[0].\
                    firstChild.data)
        else:
            self.particlesDiameter = str(particlesDiameterType)
        # Mass
        particlesMassType = \
          particlesElement.getElementsByTagName("density")[0].\
            attributes["type"].value
        if particlesMassType == "constant":
            self.particlesMass = \
              float(particlesElement.getElementsByTagName("density")[0].\
                    firstChild.data)
        else:
            self.particlesMass = str(particlesMassType)


# -----------------------------------------------------------------------------
def WriteOutput(timeIntegrator, outputFileNamePrefix):

    # Example: write slice in Z at the middle of the domain
    sliceIndex = timeIntegrator.flow.mesh.numPointsZWithGhost/2
    fields = [timeIntegrator.flow.rho, \
              timeIntegrator.flow.rhoVel, \
              timeIntegrator.flow.rhoEnergy, \
              timeIntegrator.flow.velocity, \
              timeIntegrator.flow.pressure, \
              timeIntegrator.flow.temperature]
    fieldNames = ['rho', \
                  'rhoVel', \
                  'rhoEnergy', \
                  'velocity', \
                  'pressure', \
                  'temperature']
    for field in fields:
        outputArray = field.data[:,:,sliceIndex]
        outputFileName = outputFileNamePrefix + \
                         "_" + str(timeIntegrator.timeStep).zfill(8) + \
                         "_rho"
        field.WriteSlice(2,sliceIndex,outputFileName,includeGhost=True)

    # Write output from particle distribution
    outputFileName = outputFileNamePrefix + \
                     "_" + str(timeIntegrator.timeStep).zfill(8) + \
                     "_particles.txt"
    timeIntegrator.particles.WriteOutput(outputFileName)

# -----------------------------------------------------------------------------

def Write2DArrayToMatrix(coorM, coorN, minMdx, maxMdx, minNdx, maxNdx,
                         outputArray, outputFileName):

    with open(outputFileName, 'w') as outFile:
        # Write header
        outFile.write("# x/y ")
        for ndx in range(minMdx,maxMdx):
            outFile.write(str(coorN[ndx]) +" ")
        outFile.write("\n")
        # Write contents
        for mdx in range(minMdx,maxMdx):
            outFile.write(str(coorN[mdx]) + " ")
            for ndx in range(minNdx,maxNdx):
                outFile.write(str(outputArray[mdx,ndx]) + " ")
            outFile.write("\n")

# -----------------------------------------------------------------------------


def main(argv=None):

    if argv is None:
        argv = sys.argv

    try:

#       Parse arguments and options -------------------------------------------
        requiredNumberOfArguments=2
        usage = "usage: %prog [options] arguments \n \n" + \
                "Arguments  \n" + \
                "  inputFileName[1]     \n" + \
                "  outputFileName[2]     \n"
        parser = OptionParser(usage)
        parser.add_option("-d", "--debug",
                          action="store_true", dest="debug")
        parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose")
        parser.add_option("-q", "--quiet",
                          action="store_false", dest="verbose")
        parser.add_option("-l", "--headerLines",
                          help="Dummy parameter",
                          type="int", default=0, 
                          dest="dummy")

        (options, args) = parser.parse_args()
        if len(args) != requiredNumberOfArguments:
            msg = "Incorrect number of arguments: " + \
                   str(len(args)) + " vs " + \
                   str(requiredNumberOfArguments) + " " + \
                   str(args)
            raise RuntimeError('Parsing options: %s', msg)



#       Assign arguments and options to module variables ----------------------
        dummyOption = options.dummy

        inputFileName = args[0]
        outputFileNamePrefix = args[1]

        # --------------------------------------------------------------------
        # Init
        # --------------------------------------------------------------------
        # Instantiate objects used by solver
        xmlOptions = SolverOptions(inputFileName)

        # Init spatialStencils
        spatialStencils = Stencils(xmlOptions.spatialStencilsType,
                                   xmlOptions.spatialStencilsOrder,
                                   xmlOptions.spatialStencilsSplit)

        # Init mesh
        mesh = Mesh(inputFileName, spatialStencils)

        # Init fluid
        fluid = Fluid(xmlOptions.fluidConstant,
                      xmlOptions.fluidGamma,
                      xmlOptions.fluidDynamicViscosityRef,
                      xmlOptions.fluidDynamicViscosityTemperatureRef,
                      xmlOptions.fluidPrandtl)

        # Init flow fields
        flow = Flow(inputFileName, mesh,fluid)

        # Init particles
        particles = ParticleDistribution(int(xmlOptions.particlesNumber),
                                         xmlOptions.particlesType,
                                         xmlOptions.particlesCoorX,
                                         xmlOptions.particlesCoorY,
                                         xmlOptions.particlesCoorZ,
                                         xmlOptions.particlesVelocityX,
                                         xmlOptions.particlesVelocityY,
                                         xmlOptions.particlesVelocityZ,
                                         xmlOptions.particlesDiameter,
                                         xmlOptions.particlesMass)

        # Set velocity of particles to local flow velocity
        #particles.SetVelocitiesToFlow(flow)

        # Init time integration parameters
        timeIntegrator = TimeIntegrator(inputFileName,flow,particles)

        # --------------------------------------------------------------------
        # Time advancement
        # --------------------------------------------------------------------
        timeIntegrator.timeStep = 0

        # Write output with initial data
        WriteOutput(timeIntegrator,outputFileNamePrefix)

        while timeIntegrator.simulationTime < timeIntegrator.finalTime:

            # Update time step
            timeIntegrator.timeStep += 1

            # Check for condition on maximum number of time steps
            if timeIntegrator.maxTimeStep >= 0 and \
               timeIntegrator.timeStep > timeIntegrator.maxTimeStep:
                log.info('---- Specified max time step reached; exiting.')
                break

            # Calculate deltaTime for this time step
            timeIntegrator.CalculateDeltaTime()
            log.info('---- Time step: %d, deltaTime: %f, time: %f to %f', 
                     timeIntegrator.timeStep, 
                     timeIntegrator.deltaTime,
                     timeIntegrator.simulationTime,
                     timeIntegrator.simulationTime + timeIntegrator.deltaTime)

            # Advance time step
            #log.info('AdvanceTimeStep')
            timeIntegrator.AdvanceTimeStep()

            WriteOutput(timeIntegrator,outputFileNamePrefix)

    except Exception, err:
        log.exception('Executing soleil')
        return 1
    finally:
        log.info('Cleaning up')


if __name__ == '__main__':
    sys.exit(main())

