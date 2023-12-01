
#ifndef PROFILER_HEADER
#define PROFILER_HEADER

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// profiled_bram lhsdata1a[IN_BUF_LEN];
// profiled_bram lhsdata2a[IN_BUF_LEN];
// profiled_bram lhsdata3a[IN_BUF_LEN];
// profiled_bram lhsdata4a[IN_BUF_LEN];

// struct profiled_bram {
//   ACC_DTYPE<32> lhsdata1a[IN_BUF_LEN];

// #ifndef __SYNTHESIS__
//   BufferSpace *gweightbuf_p = new BufferSpace("gweightbuf_p", GWE_BUF_LEN);

//   int capacity = GWE_BUF_LEN;
//   int write_count = 0;
//   int access_count = 0;
//   int highest_index = 0;

//   void utilisation() { gweightbuf_p->value = 0; }

//   profiled_bram::operator=(const profiled_bram &rhs) const {
//     write_count++;
//     return lhsdata1a[] = rhs.lhsdata1a;
//   }

//   profiled_bram::operator[](int index) const {
//     write_count++;
//     return lhsdata1a[index];
//   }

// #endif
// }

// struct profiled_bram<32>
//     lhsdata4a[1] = 5;

enum MetricTypes { TClockCycles, TDataCount, TBufferSpace, TDataCountArray };

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