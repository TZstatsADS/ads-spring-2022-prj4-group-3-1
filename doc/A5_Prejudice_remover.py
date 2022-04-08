# The code for PrejudiceRemover is a modification of, and based on, the
# implementation of Kamishima Algorithm by fairness-comparison.

# Reference: https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/inprocessing/prejudice_remover.py

import numpy as np
import pandas as pd
import tempfile
import os
import subprocess

from aif360.algorithms import Transformer


class PrejudiceRemover(Transformer):

    def __init__(self, eta=1.0, sensitive_attr="", class_attr=""):

        super(PrejudiceRemover, self).__init__(eta=eta,
            sensitive_attr=sensitive_attr, class_attr=class_attr)
        self.eta = eta
        self.sensitive_attr = sensitive_attr
        self.class_attr = class_attr

    def _create_file_in_kamishima_format(self, df, class_attr,
                                         positive_class_val, sensitive_attrs,
                                         single_sensitive, privileged_vals):

        x = []
        for col in df:
            if col != class_attr and col not in sensitive_attrs:
                x.append(np.array(df[col].values, dtype=np.float64))
        x.append(np.array(single_sensitive.isin(privileged_vals),
                          dtype=np.float64))
        x.append(np.array(df[class_attr] == positive_class_val,
                          dtype=np.float64))

        fd, name = tempfile.mkstemp()
        os.close(fd)
        np.savetxt(name, np.array(x).T)
        return name

    def fit(self, dataset):

        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        train_df = pd.DataFrame(data=data, columns=columns)

        all_sensitive_attributes = dataset.protected_attribute_names

        if not self.sensitive_attr:
            self.sensitive_attr = all_sensitive_attributes[0]
        self.sensitive_ind = all_sensitive_attributes.index(self.sensitive_attr)

        sens_df = pd.Series(dataset.protected_attributes[:, self.sensitive_ind],
                            name=self.sensitive_attr)

        if not self.class_attr:
            self.class_attr = dataset.label_names[0]

        fd, model_name = tempfile.mkstemp()
        os.close(fd)
        train_name = self._create_file_in_kamishima_format(train_df,
                self.class_attr, dataset.favorable_label,
                all_sensitive_attributes, sens_df,
                dataset.privileged_protected_attributes[self.sensitive_ind])

        k_path = os.path.dirname(os.path.abspath(__file__))
        train_pr = os.path.join(k_path, 'kamfadm-2012ecmlpkdd', 'train_pr.py')

        subprocess.call(['python', train_pr,
                         '-e', str(self.eta),
                         '-i', train_name,
                         '-o', model_name,
                         '--quiet'])
        os.unlink(train_name)

        self.model_name = model_name

        return self

    def predict(self, dataset):

        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        test_df = pd.DataFrame(data=data, columns=columns)
        sens_df = pd.Series(dataset.protected_attributes[:, self.sensitive_ind],
                            name=self.sensitive_attr)

        fd, output_name = tempfile.mkstemp()
        os.close(fd)

        test_name = self._create_file_in_kamishima_format(test_df,
                self.class_attr, dataset.favorable_label,
                dataset.protected_attribute_names, sens_df,
                dataset.privileged_protected_attributes[self.sensitive_ind])


        k_path = os.path.dirname(os.path.abspath(__file__))
        predict_lr = os.path.join(k_path, 'kamfadm-2012ecmlpkdd', 'predict_lr.py')

        subprocess.call(['python', predict_lr,
                         '-i', test_name,
                         '-m', self.model_name,
                         '-o', output_name,
                         '--quiet'])
        os.unlink(test_name)
        m = np.loadtxt(output_name)
        os.unlink(output_name)

        pred_dataset = dataset.copy()

        pred_dataset.labels = m[:, [1]]
        pred_dataset.scores = m[:, [4]]

        return pred_dataset
