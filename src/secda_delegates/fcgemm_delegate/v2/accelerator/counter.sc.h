#ifndef __SYNTHESIS__
void ACCNAME::Simulation_Profiler() {
  wait();
  while (1) {
    total_cycles->value++;
    wait();
  }
}
#endif