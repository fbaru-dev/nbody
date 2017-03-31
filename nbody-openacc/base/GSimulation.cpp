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
    particles[i].pos[0] = -1.0f + 2.0f * r / max; 
    particles[i].pos[1] = -1.0f + 2.0f * r / max;  
    particles[i].pos[2] = -1.0f + 2.0f * r / max;     
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
    particles[i].vel[0] = -1.0e-4 + 2.0f * r / max * 1.0e-4f;  
    particles[i].vel[1] = -1.0e-4 + 2.0f * r / max * 1.0e-4f; 
    particles[i].vel[2] = -1.0e-4 + 2.0f * r / max * 1.0e-4f; 
  }
}

void GSimulation :: init_acc() 
{
  for(int i=0; i<get_npart(); ++i)
  {
    particles[i].acc[0] = 0; 
    particles[i].acc[1] = 0;
    particles[i].acc[2] = 0;
  }
}

void GSimulation :: init_mass() 
{
//   std::random_device rd;	//random number generator
//   std::mt19937 gen(42);  
//   std::uniform_int_distribution<int> unif_d(1,R_MAX);
//   
//   real_type n = static_cast<real_type> (get_npart());
//   real_type max = static_cast<real_type> (R_MAX);
// 
//   for(int i=0; i<get_npart(); ++i)
//   {
//     real_type r = static_cast<real_type> ( unif_d(gen) );
//     particles[i].mass = n + n * r / max; 
//   }
  int gen = 42;
  srand(gen);
  real_type n   = static_cast<real_type> (get_npart());
  real_type max = static_cast<real_type> (RAND_MAX);

  for(int i=0; i<get_npart(); ++i)
  {
    real_type r = static_cast<real_type>(rand()) / static_cast<real_type>(RAND_MAX); 
    r = (max - 1.0f) * r + 1.0f;
    particles[i].mass =  n + n * r / max; 
  }
}

void GSimulation :: start() 
{
  real_type energy;
  real_type dt = get_tstep();
  int n = get_npart();
  int i,j;
  
  //allocate particles
  particles = new Particle[n];
 
  init_pos();	
  init_vel();
  init_acc();
  init_mass();
  
  print_header();
  
  _totTime = 0.; 
  
  const double softeningSquared = 0.01*0.01;
  // prevents explosion in the case the particles are really close to each other 
  const double G = 6.67259e-11;
  
  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;
  
  double gflops = 1e-9 * ( (11. + 18. ) * double( (n*n-1) ) +  double(n) * 19. );
  double av=0.0, dev=0.0;
  int nf = 0;
  
  const double t0 = time.start();

  #pragma acc enter data copyin(this[0:1])
  #pragma acc enter data copyin(particles[0:n])
  
  for (int s=1; s<=get_nsteps(); ++s)
  {   
    ts0 += time.start();
   #pragma acc data present(this[0:1],particles[0:n]) 
   {   //Start parallel region
    #pragma acc parallel loop 
    for (i = 0; i < n; i++)// update acceleration
    {
      real_type ax_i = particles[i].acc[0];
      real_type ay_i = particles[i].acc[1];
      real_type az_i = particles[i].acc[2];
      for (j = 0; j < n; j++)
      {
	if (j != i)
	{
	  real_type dx, dy, dz;
	  real_type distanceSqr = 0.0f;
	  real_type distanceInv = 0.0f;
		  
	  dx = particles[j].pos[0] - particles[i].pos[0];	//1flop
	  dy = particles[j].pos[1] - particles[i].pos[1];	//1flop	
	  dz = particles[j].pos[2] - particles[i].pos[2];	//1flop
	
	  distanceSqr = dx*dx + dy*dy + dz*dz + softeningSquared;	//6flops
	  distanceInv = 1.0 / sqrt(distanceSqr);			//1div+1sqrt
		  
	  ax_i += dx * G * particles[j].mass * distanceInv * distanceInv * distanceInv;	//6flops
	  ay_i += dy * G * particles[j].mass * distanceInv * distanceInv * distanceInv;	//6flops
	  az_i += dz * G * particles[j].mass * distanceInv * distanceInv * distanceInv;	//6flops
	}
      }
      particles[i].acc[0] = ax_i;
      particles[i].acc[1] = ay_i;
      particles[i].acc[2] = az_i;
    }
    energy = 0;
    #pragma acc parallel loop reduction(+:energy)
    for (i = 0; i < n; ++i)// update position and velocity
    {
      particles[i].vel[0] += particles[i].acc[0] * dt;	//2flops
      particles[i].vel[1] += particles[i].acc[1] * dt;	//2flops
      particles[i].vel[2] += particles[i].acc[2] * dt;	//2flops
	 
      particles[i].pos[0] += particles[i].vel[0] * dt;	//2flops
      particles[i].pos[1] += particles[i].vel[1] * dt;	//2flops
      particles[i].pos[2] += particles[i].vel[2] * dt;	//2flops

      particles[i].acc[0] = 0.;
      particles[i].acc[1] = 0.;
      particles[i].acc[2] = 0.;
	
      energy += particles[i].mass * (
		particles[i].vel[0]*particles[i].vel[0] + 
                particles[i].vel[1]*particles[i].vel[1] +
                particles[i].vel[2]*particles[i].vel[2]); //7flops
    }
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
  delete particles;
}
