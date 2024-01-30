
#ifndef PROFILER_HEADER
#define PROFILER_HEADER

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef ACC_PROFILE
#define prf_start(N) auto start##N = chrono::high_resolution_clock::now();
#define prf_end(N, X)                                                          \
  auto end##N = chrono::high_resolution_clock::now();                          \
  X += end##N - start##N;
#else
#define prf_start(N)
#define prf_end(N, X)
#endif

using namespace std;
using namespace std::chrono;
#define prf_out(TSCALE, X)                                                     \
  cerr << #X << ": " << duration_cast<TSCALE>(X).count() << endl;

#define prf_file_out(TSCALE, X, file)                                          \
  file << #X << "," << duration_cast<TSCALE>(X).count() << endl;

typedef duration<long long int, std::ratio<1, 1000000000>> duration_ns;

#ifdef SYSC
#define SYSC_ON(X) X
#else
#define SYSC_ON(X)
#endif

enum MetricTypes {
  TClockCycles,
  TDataCount,
  TBufferSpace,
  TDataCountArray,
  TSignalTrack
};

class Metric {
public:
  string name;
  int value;
  MetricTypes type;
  // Metric(string _name,int value,  MetricTypes _type);
};

class ClockCycles : public Metric {
public:
  ClockCycles(string _name);
  ClockCycles(string _name, bool _resetOnSave);

  int readCount();
  bool resetOnSave;
};

class SignalTrack : public Metric {
public:
  vector<int> values;
  SignalTrack(string _name);
  SignalTrack(string _name, bool _resetOnSave);

  int readCount();
  void increment(int);
  bool resetOnSave;
};

class DataCount : public Metric {
public:
  DataCount(string _name);
  bool resetOnSave;
};

class DataCountArray : public Metric {
public:
  DataCountArray(string _name, int size);
  int *array;
  bool resetOnSave;
};

class BufferSpace : public Metric {

public:
  BufferSpace(string _name, int _total);
  int total;
  bool resetOnSave = true;
};

class Profile {
public:
  vector<Metric> base_metrics;

  void initProfile();

  void addMetric(Metric);

  void updateMetric(Metric);

  void incrementMetric(string, int);

  // Creates copy of the profiled metrics as a record
  void saveProfile(vector<Metric *>);

  void saveBlank(vector<Metric *>);

  // Creates CSV of all saved Records
  void saveCSVRecords(string);

private:
  vector<vector<Metric>> records;
  vector<Metric> model_record;
};

template <typename T>
void saveMatrixCSV(string filename, T *matrix, int rows, int cols) {
  ofstream file;
  file.open(filename);
  int index = 0;
  for (int c = 0; c < rows; c++) {
    file << endl;
    for (int r = 0; r < cols; r++) {
      file << (int)matrix[index] << ",";
      index++;
    }
  }
  file.close();
};

template <typename T>
void printMatrixCSV(T *matrix, int rows, int cols) {
  int index = 0;
  for (int c = 0; c < rows; c++) {
    cerr << endl;
    for (int r = 0; r < cols; r++) {
      cerr << (int)matrix[index] << ",";
      index++;
    }
  }
};

#endif // PROFILER_HEADER