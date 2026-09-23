#ifndef AMSS_PROFILE_H
#define AMSS_PROFILE_H

#ifdef AMSS_ENABLE_PROFILE
namespace amss_profile
{
  enum Stage
  {
    INITIALIZE = 0,
    EVOLVE,
    RECURSIVE_STEP,
    LEVEL_STEP,
    RESTRICT_PROLONG,
    ANALYSIS,
    PSI4,
    CONSTRAINT,
    MPI_TRANSFER,
    MPI_SYNC,
    WAVE_SURFACE,
    WAVE_INTERP,
    WAVE_INTEGRATE,
    ADM_SURFACE,
    ADM_INTERP,
    ADM_INTEGRATE,
    STAGE_COUNT
  };

  class Scope
  {
  public:
    explicit Scope(Stage stage);
    ~Scope();
    void stop();
  private:
    Stage stage_;
    double started_;
  };

  void report();
}
#endif

#endif
