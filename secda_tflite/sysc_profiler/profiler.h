
#ifndef PROFILER_HEADER
#define PROFILER_HEADER

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

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
  DataCountArray(string _name,int size);
  int* array;
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

  // Creates CSV of all saved Records
  void saveCSVRecords(string);

private:
  vector<vector<Metric>> records;
  vector<Metric> model_record;
};

#endif // PROFILER_HEADER