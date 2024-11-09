import numpy as np
import PerformanceMetricHelper as metric
import importlib

importlib.reload(metric)

class Metrics:
    def __init__(self) :
        self.metrics = {
            'accuracy':[],
            'sensitivity':[],
            'specificity':[],
            'precision':[],
            'f1-score':[],
            'auroc':[],
            'aupr':[]
        }

    def update_the_metrics(self, y_true, y_pred):
        self.metrics['accuracy'].append(metric.custom_accuracy(y_true, y_pred))
        self.metrics['sensitivity'].append(metric.custom_sensitivity(y_true, y_pred))
        self.metrics['specificity'].append(metric.custom_specificity(y_true, y_pred))
        self.metrics['precision'].append(metric.custom_precision(y_true, y_pred))
        self.metrics['f1-score'].append(metric.f1_score_(y_true, y_pred))
        self.metrics['auroc'].append(metric.auroc(y_true, y_pred))
        self.metrics['aupr'].append(metric.aupr(y_true, y_pred))

    def calculate_mean_stddev(self):
        for metric_name, values in self.metrics.items():
            if values:
                mean = np.mean(values)
                stddev = np.std(values)
                self.metrics[metric_name].clear()
                self.metrics[metric_name].append(mean+stddev)

    def print_(self):
        for metric_name, values in self.metrics.items():
            print(f'{metric_name} : {values}')

    def get_metrics_summary(self):
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = values[0]
        return summary
