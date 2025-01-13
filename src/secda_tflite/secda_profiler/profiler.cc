
#include "profiler.h"

ClockCycles::ClockCycles(string _name) {
  name = _name;
  value = 0;
  type = TClockCycles;
}

ClockCycles::ClockCycles(string _name, bool _resetOnSave) {
  name = _name;
  value = 0;
  type = TClockCycles;
  resetOnSave = _resetOnSave;
}

int SignalTrack::readCount() {
  if (resetOnSave) value = 0;
  return value;
}

SignalTrack::SignalTrack(string _name) {
  name = _name;
  value = 0;
  type = TSignalTrack;
}

void SignalTrack::increment(int val) {
  if (values.size() < val + 1) values.resize(val + 1);
  values[val]++;
}

SignalTrack::SignalTrack(string _name, bool _resetOnSave) {
  name = _name;
  value = 0;
  type = TSignalTrack;
  resetOnSave = _resetOnSave;
}

int ClockCycles::readCount() {
  if (resetOnSave) value = 0;
  return value;
}

DataCount::DataCount(string _name) {
  name = _name;
  type = TDataCount;
  value = 0;
  resetOnSave = true;
}

DataCountArray::DataCountArray(string _name, int size) {
  name = _name;
  type = TDataCountArray;
  value = size;
  array = new int[size]();
  resetOnSave = true;
}

BufferSpace::BufferSpace(string _name, int _total) {
  name = _name;
  value = 0;
  type = TBufferSpace;
  total = _total;
}

// Currently this does not check if the metric profiles match
void Profile::saveProfile(vector<Metric *> captured_metrics) {
#ifdef SYSC
  vector<Metric> newRecord;

  for (int i = 0; i < captured_metrics.size(); i++) {
    if (captured_metrics[i]->type == TClockCycles) {
      ClockCycles *capped_metric =
          reinterpret_cast<ClockCycles *>(captured_metrics[i]);
      ClockCycles temp(capped_metric->name);
      temp.value = capped_metric->value;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TDataCount) {
      DataCount *capped_metric =
          reinterpret_cast<DataCount *>(captured_metrics[i]);
      DataCount temp(capped_metric->name);
      temp.value = capped_metric->value;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TBufferSpace) {
      BufferSpace *capped_metric =
          reinterpret_cast<BufferSpace *>(captured_metrics[i]);
      BufferSpace temp(capped_metric->name, capped_metric->total);
      temp.value = capped_metric->value;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TDataCountArray) {
      DataCountArray *capped_metric =
          reinterpret_cast<DataCountArray *>(captured_metrics[i]);
      for (int l = 0; l < capped_metric->value; l++) {
        DataCount temp(capped_metric->name + to_string(l));
        temp.value = capped_metric->array[l];
        if (capped_metric->resetOnSave) capped_metric->array[l] = 0;
        newRecord.push_back(temp);
      }
    } else if (captured_metrics[i]->type == TSignalTrack) {
      SignalTrack *capped_metric =
          reinterpret_cast<SignalTrack *>(captured_metrics[i]);
      for (int l = 0; l < capped_metric->values.size(); l++) {
        if (capped_metric->values[l] > 0) {
          ClockCycles temp(capped_metric->name + "_" + to_string(l));
          temp.value = capped_metric->values[l];
          newRecord.push_back(temp);
        }
        if (capped_metric->resetOnSave) capped_metric->values[l] = 0;
      }
      if (capped_metric->resetOnSave) capped_metric->value = 0;
    }
  }
  records.push_back(newRecord);
#endif
}

void Profile::saveBlank(vector<Metric *> captured_metrics) {
  vector<Metric> newRecord;

  for (int i = 0; i < captured_metrics.size(); i++) {
    if (captured_metrics[i]->type == TClockCycles) {
      ClockCycles *capped_metric =
          reinterpret_cast<ClockCycles *>(captured_metrics[i]);
      ClockCycles temp(capped_metric->name);
      temp.value = -1;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TDataCount) {
      DataCount *capped_metric =
          reinterpret_cast<DataCount *>(captured_metrics[i]);
      DataCount temp(capped_metric->name);
      temp.value = -1;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TBufferSpace) {
      BufferSpace *capped_metric =
          reinterpret_cast<BufferSpace *>(captured_metrics[i]);
      BufferSpace temp(capped_metric->name, capped_metric->total);
      temp.value = -1;
      if (capped_metric->resetOnSave) capped_metric->value = 0;
      newRecord.push_back(temp);
    } else if (captured_metrics[i]->type == TDataCountArray) {
      DataCountArray *capped_metric =
          reinterpret_cast<DataCountArray *>(captured_metrics[i]);
      for (int l = 0; l < capped_metric->value; l++) {
        DataCount temp(capped_metric->name + to_string(l));
        temp.value = -1;
        if (capped_metric->resetOnSave) capped_metric->array[l] = 0;
        newRecord.push_back(temp);
      }
    }
  }
  records.push_back(newRecord);
}

void Profile::addMetric(Metric m) { model_record.push_back(m); }

void Profile::updateMetric(Metric m) {
  for (int i = 0; i < model_record.size(); i++) {
    if (m.name == model_record[i].name) {
      model_record[i] = m;
    }
  }
}

void Profile::incrementMetric(string name, int value) {
  for (int i = 0; i < model_record.size(); i++) {
    if (name == model_record[i].name) {
      model_record[i].value += value;
    }
  }
}

void Profile::saveCSVRecords(string filename) {
#ifdef SYSC

  if (records.size() == 0) return;
  mkdir(".data", 0777);
  ofstream per_sim_file;
  per_sim_file.open(filename + ".csv");

  for (int i = 0; i < base_metrics.size(); i++) {
    per_sim_file << base_metrics[i].name;
    if (i + 1 != base_metrics.size()) per_sim_file << ",";
    else per_sim_file << endl;
  }

  for (int i = 0; i < records[0].size(); i++) {
    per_sim_file << records[0][i].name;
    if (i + 1 != records[0].size()) per_sim_file << ",";
    else per_sim_file << endl;
  }

  std::vector<std::string> metric_names;
  for (int r = 0; r < records.size(); r++) {
    for (int m = 0; m < records[r].size(); m++) {
      if (std::find(metric_names.begin(), metric_names.end(),
                    records[r][m].name) == metric_names.end()) {
        metric_names.push_back(records[r][m].name);
      }
      // push  records[r][m].name to metric_names if not already there
      // bool add = true;
      // for (int i = 0; i < metric_names.size(); i++) {
      //   if (metric_names[i] == records[r][m].name) {
      //     add = false;
      //     break;
      //   }
      // }
      // if (add) metric_names.push_back(records[r][m].name);
    }
  }

  // create csv columns
  for (int r = 0; r < records.size(); r++) {
    for (int m = 0; m < metric_names.size(); m++) {
      bool found = false;
      for (int i = 0; i < records[r].size(); i++) {
        if (records[r][i].name == metric_names[m]) {
          per_sim_file << records[r][i].value;
          if (m + 1 != metric_names.size()) per_sim_file << ",";
          else per_sim_file << endl;
          found = true;
        }
      }
      if (!found) {
        per_sim_file << "0";
        if (m + 1 != metric_names.size()) per_sim_file << ",";
        else per_sim_file << endl;
      }
    }
  }
  per_sim_file.close();

  // ofstream per_model_file;
  // per_model_file.open(filename + "_model.csv");
  // for (int i = 0; i < model_record.size(); i++) {
  //   per_model_file << model_record[i].name;
  //   if (i + 1 != model_record.size()) per_model_file << ",";
  //   else per_model_file << endl;
  // }

  // for (int m = 0; m < model_record.size(); m++) {
  //   per_model_file << model_record[m].value;
  //   if (m + 1 != model_record.size()) per_model_file << ", ";
  //   else per_model_file << endl;
  // }
  // per_model_file.close();
#endif
}