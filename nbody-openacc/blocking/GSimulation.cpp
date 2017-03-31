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

GSimulation :: GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(2000); 
  set_nsteps(500);
  set_tstep(0.1); 
  set_sfreq(50);
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
  int i,j;
  
  //allocate particles
  //const int alignment = 32;
  //particles = (ParticleSoA*) _mm_malloc(sizeof(ParticleSoA),alignment);

  //particles->pos_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->pos_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->pos_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->vel_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->vel_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->vel_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->acc_x = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->acc_y = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->acc_z = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);
  //particles->mass  = (real_type*) _mm_malloc(n*sizeof(real_type),alignment);

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
  
  const float softeningSquared = 0.01f*0.01f;
  const float G = 6.67259e-11f;
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  
  double gflops = 1e-9 * ( (11. + 18. ) * double( (n*n-1) ) +  double(n) * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const int size_tile = 8;

  const double t0 = time.start();
  
  #pragma acc enter data copyin(this[0:1],particles[0:1])
  #pragma acc enter data copyin(particles->pos_x[0:n], particles->pos_y[0:n], particles->pos_z[0:n])
  #pragma acc enter data copyin(particles->vel_x[0:n], particles->vel_y[0:n], particles->vel_z[0:n])
  #pragma acc enter data copyin(particles->acc_x[0:n], particles->acc_y[0:n], particles->acc_z[0:n])
  #pragma acc enter data copyin(particles->mass[0:n])
  
  for (int s=1; s<=get_nsteps(); ++s)
  {   
   ts0 += time.start();
   
   #pragma acc data present(this[0:1],particles[0:1], particles->pos_x[0:n], particles->pos_y[0:n], particles->pos_z[0:n]) \
                              present(particles->vel_x[0:n], particles->vel_y[0:n], particles->vel_z[0:n]) \
                              present(particles->acc_x[0:n], particles->acc_y[0:n], particles->acc_z[0:n]) \
                              present(particles->mass[0:n])
  {  // start of parallel region
   #pragma acc parallel loop 
   for (int ii = 0; ii < n; ii += size_tile )// update acceleration
   {
     real_type acc_xtile[size_tile];
     real_type acc_ytile[size_tile] ;
     real_type acc_ztile[size_tile];
     acc_xtile[:] = 0.0f;
     acc_ytile[:] = 0.0f;
     acc_ztile[:] = 0.0f; 
     for (j = 0; j < n; j++)
     {
       real_type m = particles->mass[j];
       real_type px_j = particles->pos_x[j];
       real_type py_j = particles->pos_y[j];
       real_type pz_j = particles->pos_z[j];
       for (int i = ii; i < ii + size_tile; i++)
       {
// 	 if (j != i)
//          {
	   real_type dx, dy, dz;
	   real_type distanceSqr = 0.0f;
	   real_type distanceInv = 0.0f;
		  
	   dx = px_j -  particles->pos_x[i];	//1flop
	   dy = py_j -  particles->pos_y[i];	//1flop	
	   dz = pz_j -  particles->pos_z[i];	//1flop
	
 	   distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
 	   distanceInv = 1.0f / sqrtf(distanceSqr);			//1div+1sqrt

	   acc_xtile[i-ii] += dx * G * m * distanceInv * distanceInv * distanceInv;	//6flops
	   acc_ytile[i-ii] += dy * G * m * distanceInv * distanceInv * distanceInv;	//6flops
	   acc_ztile[i-ii] += dz * G * m * distanceInv * distanceInv * distanceInv;
// 	 }
      }
     }
     particles->acc_x[ii:size_tile] = acc_xtile[0:size_tile];
     particles->acc_y[ii:size_tile] = acc_ytile[0:size_tile];
     particles->acc_z[ii:size_tile] = acc_ztile[0:size_tile];
  }
   
   energy = 0;
   #pragma acc parallel loop reduction(+:energy)
   for (i = 0; i < n; ++i)// update position
   {
     particles->vel_x[i] += particles->acc_x[i] * dt; //2flops
     particles->vel_y[i] += particles->acc_y[i] * dt; //2flops
     particles->vel_z[i] += particles->acc_z[i] * dt; //2flops
	  
     particles->pos_x[i] += particles->vel_x[i] * dt; //2flops
     particles->pos_y[i] += particles->vel_y[i] * dt; //2flops
     particles->pos_z[i] += particles->vel_z[i] * dt; //2flops

     particles->acc_x[i] = 0.;
     particles->acc_y[i] = 0.;
     particles->acc_z[i] = 0.;
	
     energy += particles->mass[i] * (
	       particles->vel_x[i]*particles->vel_x[i] + 
               particles->vel_y[i]*particles->vel_y[i] +
               particles->vel_z[i]*particles->vel_z[i]); //7flops
   }
  }  // end of parallel region
 
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
//  _mm_free(particles->pos_x);
//  _mm_free(particles->pos_y);
//  _mm_free(particles->pos_z);
//  _mm_free(particles->vel_x);
//  _mm_free(particles->vel_y);
//  _mm_free(particles->vel_z);
//  _mm_free(particles->acc_x);
//  _mm_free(particles->acc_y);
//  _mm_free(particles->acc_z);
//  _mm_free(particles->mass);
//  _mm_free(particles);
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
