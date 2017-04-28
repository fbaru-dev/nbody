/*
    This file is part of the example codes which have been used
    for the "Code Optmization Workshop".
    
    Copyright (C) 2016  Fabio Baruffa <fbaru-dev@gmail.com>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "GSimulation.hpp"
#include "cpu_time.hpp"
#include <cuda.h>

__global__ void computeVel(real_type *d_px,real_type *d_py,real_type *d_pz,real_type *d_vx,real_type *d_vy,real_type *d_vz,
real_type *d_m, real_type dt, int n);


GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(2000); 
  set_nsteps(10);
  set_tstep(0.1); 
  set_sfreq(1);
}

void GSimulation :: set_number_of_particles(int N)  
{
  set_npart(N);
}

void GSimulation :: set_number_of_steps(int N)  
{
  set_nsteps(N);
}

void GSimulation :: init_pos()  
{
  int gen = 42; 
  srand(gen);
  real_type max = static_cast<real_type> ( R_MAX );
  
  for(int i=0; i<get_npart(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->pos_x[i] = -1.0f + 2.0f * r / max; 
    particles->pos_y[i] = -1.0f + 2.0f * r / max;  
    particles->pos_z[i] = -1.0f + 2.0f * r / max;     
  }
}

void GSimulation :: init_vel()  
{
  int gen = 42;
  srand(gen);
  real_type max = static_cast<real_type> (RAND_MAX);

  for(int i=0; i<get_npart(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->vel_x[i] = -1.0e-4 + 2.0f * r / max * 1.0e-4f;  
    particles->vel_y[i] = -1.0e-4 + 2.0f * r / max * 1.0e-4f; 
    particles->vel_z[i] = -1.0e-4 + 2.0f * r / max * 1.0e-4f; 
  }
}

void GSimulation :: init_acc() 
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles->acc_x[i] = 0; 
    particles->acc_y[i] = 0;
    particles->acc_z[i] = 0;
  }
}

void GSimulation :: init_mass() 
{
  int gen = 42;
  srand(gen);
  real_type n   = static_cast<real_type> (get_npart());
  real_type max = static_cast<real_type> (RAND_MAX);

  for(int i=0; i<get_npart(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles->mass[i] =  n + n * r / max; 
  }
}

void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();

  particles = new ParticleSoA;
  
  particles->pos_x = new real_type[n];
  particles->pos_y = new real_type[n];
  particles->pos_z = new real_type[n];
  particles->vel_x = new real_type[n];
  particles->vel_y = new real_type[n];
  particles->vel_z = new real_type[n];
  particles->acc_x = new real_type[n];
  particles->acc_y = new real_type[n];
  particles->acc_z = new real_type[n];
  particles->mass  = new real_type[n]; 
  
  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
 
  _totTime = 0.; 
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  double nd = double(n);
  double gflops = 1e-9 * ( (11. + 18. ) * nd*(nd-1.0)  +  nd * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const double t0 = time.start();
  
  real_type *d_px, *d_py, *d_pz;
  real_type *d_vx, *d_vy, *d_vz;
  real_type *d_m;
  cudaMalloc(&d_px, n*sizeof(real_type));
  cudaMalloc(&d_py, n*sizeof(real_type));
  cudaMalloc(&d_pz, n*sizeof(real_type));
  cudaMalloc(&d_vx, n*sizeof(real_type));
  cudaMalloc(&d_vy, n*sizeof(real_type));
  cudaMalloc(&d_vz, n*sizeof(real_type));
  cudaMalloc(&d_m,  n*sizeof(real_type));
  int blockSize = 256;
  int nBlocks = (n + blockSize - 1) / blockSize; 
  
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start();
   
   cudaMemcpy(d_px, particles->pos_x, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_py, particles->pos_y, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_pz, particles->pos_z, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_vx, particles->vel_x, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_vy, particles->vel_y, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_vz, particles->vel_z, n*sizeof(real_type), cudaMemcpyHostToDevice);
   cudaMemcpy(d_m,  particles->mass, n*sizeof(real_type), cudaMemcpyHostToDevice);

   computeVel <<< nBlocks, blockSize >>> (d_px,d_py,d_pz,d_vx,d_vy,d_vz,d_m, dt, n);

   cudaMemcpy(particles->vel_x,d_vx, n*sizeof(real_type), cudaMemcpyDeviceToHost);
   cudaMemcpy(particles->vel_y,d_vy, n*sizeof(real_type), cudaMemcpyDeviceToHost);
   cudaMemcpy(particles->vel_z,d_vz, n*sizeof(real_type), cudaMemcpyDeviceToHost);

   energy = 0;
   for (int i = 0; i < n; ++i)// update position
   {
     particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
     particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
     particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

     energy += particles->mass[i] * (
               particles->vel_x[i]*particles->vel_x[i] +
               particles->vel_y[i]*particles->vel_y[i] +
               particles->vel_z[i]*particles->vel_z[i]); //7flops
   }
 
    _kenergy = 0.5 * energy; 
    
    ts1 += time.stop();
    if(!(s%get_sfreq()) ) 
    {
      nf += 1;      
      std::cout << " " 
		<<  std::left << std::setw(8)  << s
		<<  std::left << std::setprecision(5) << std::setw(8)  << s*get_tstep()
		<<  std::left << std::setprecision(5) << std::setw(12) << _kenergy
		<<  std::left << std::setprecision(5) << std::setw(12) << (ts1 - ts0)
		<<  std::left << std::setprecision(5) << std::setw(12) << gflops*get_sfreq()/(ts1 - ts0)
		<<  std::endl;
      if(nf > 2) 
      {
	av  += gflops*get_sfreq()/(ts1 - ts0);
	dev += gflops*get_sfreq()*gflops*get_sfreq()/((ts1-ts0)*(ts1-ts0));
      }
      
      ts0 = 0;
      ts1 = 0;
    }
  
  } //end of the time step loop
	
	
  const double t1 = time.stop();
  _totTime  = (t1-t0);
  _totFlops = gflops*get_nsteps();
  
  av/=(double)(nf-2);
  dev=sqrt(dev/(double)(nf-2)-av*av);
  
  int nthreads=1;
  #pragma omp parallel
  nthreads=omp_get_num_threads(); 

  std::cout << std::endl;
  std::cout << "# Number Threads     : " << nthreads << std::endl;	   
  std::cout << "# Total Time (s)     : " << _totTime << std::endl;
  std::cout << "# Average Perfomance : " << av << " +- " <<  dev << std::endl;
  std::cout << "===============================" << std::endl;

}


void GSimulation :: print_header()
{
	    
  std::cout << " nPart = " << get_npart()  << "; " 
	    << "nSteps = " << get_nsteps() << "; " 
	    << "dt = "     << get_tstep()  << std::endl;
	    
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " " 
	    <<  std::left << std::setw(8)  << "s"
	    <<  std::left << std::setw(8)  << "dt"
	    <<  std::left << std::setw(12) << "kenergy"
	    <<  std::left << std::setw(12) << "time (s)"
	    <<  std::left << std::setw(12) << "GFlops"
	    <<  std::endl;
  std::cout << "------------------------------------------------" << std::endl;


}

GSimulation :: ~GSimulation()
{
  delete [] particles->pos_x;
  delete [] particles->pos_y;
  delete [] particles->pos_z;
  delete [] particles->vel_x;
  delete [] particles->vel_y;
  delete [] particles->vel_z;
  delete [] particles->acc_x;
  delete [] particles->acc_y;
  delete [] particles->acc_z;
  delete [] particles->mass;
  delete particles;
}

__global__ void computeVel(real_type *d_px,real_type *d_py,real_type *d_pz,real_type *d_vx,real_type *d_vy,real_type *d_vz,real_type *d_m, real_type dt, int n)
{
   const float softeningSquared = 0.01f*0.01f;
   const float G = 6.67259e-11f;

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < n)
   {
     real_type ax_i = 0.0f;
     real_type ay_i = 0.0f;
     real_type az_i = 0.0f;
     for (int j = 0; j < n; j++)
     {
         real_type dx = d_px[j] - d_px[i];      //1flop
         real_type dy = d_py[j] - d_py[i];      //1flop 
         real_type dz = d_pz[j] - d_pz[i];      //1flop

         real_type distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;      //6flops
         real_type distanceInv = 1.0f / sqrtf(distanceSqr);                     //1div+1sqrt
         //real_type distanceInv = rsqrt(distanceSqr);
         ax_i += dx * G * d_m[j] * distanceInv * distanceInv * distanceInv; //6flops
         ay_i += dy * G * d_m[j] * distanceInv * distanceInv * distanceInv; //6flops
         az_i += dz * G * d_m[j] * distanceInv * distanceInv * distanceInv; //6flops
     }
     d_vx[i] += ax_i * dt; //2flops
     d_vy[i] += ay_i * dt; //2flops
     d_vz[i] += az_i * dt; //2flops   
  }
}

