#include "amss_profile.h"

#ifdef AMSS_ENABLE_PROFILE
#include <mpi.h>
#include <iostream>
#include <iomanip>

namespace
{
  double totals[amss_profile::STAGE_COUNT] = {0.0};
  unsigned long calls[amss_profile::STAGE_COUNT] = {0};
  const char *stage_names[amss_profile::STAGE_COUNT] = {
    "initialize", "evolve", "recursive_step", "level_step",
    "restrict_prolong", "analysis", "psi4", "constraint",
    "mpi_transfer", "mpi_sync",
    "wave_surface", "wave_interp", "wave_integrate",
    "adm_surface", "adm_interp", "adm_integrate"
  };
}

amss_profile::Scope::Scope(Stage stage) : stage_(stage), started_(MPI_Wtime())
{
}

amss_profile::Scope::~Scope()
{
  stop();
}

void amss_profile::Scope::stop()
{
  if (started_ < 0.0)
    return;
  totals[stage_] += MPI_Wtime() - started_;
  calls[stage_] += 1;
  started_ = -1.0;
}

void amss_profile::report()
{
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized)
    return;

  int rank = 0, nprocs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  for (int stage = 0; stage < STAGE_COUNT; ++stage)
  {
    double total_sum = 0.0, total_min = 0.0, total_max = 0.0;
    unsigned long calls_max = 0;
    MPI_Reduce(&totals[stage], &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totals[stage], &total_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&totals[stage], &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&calls[stage], &calls_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0 && calls_max != 0)
    {
      std::cout << " AMSS_PROFILE stage=" << stage_names[stage]
                << " avg_rank_s=" << std::fixed << std::setprecision(6)
                << (total_sum / nprocs)
                << " min_rank_s=" << total_min
                << " max_rank_s=" << total_max
                << " max_calls=" << calls_max << std::endl;
    }
  }
}
#endif
