import pandas as pd

file_path = 'IBM_Debater_(R)_EviConv-ACL-2019.v1/train.csv'

data = pd.read_csv(file_path)
filtered_data = data[data['evidence_1_stance'] != data['evidence_2_stance']]
filtered_data = filtered_data[['topic', 'evidence_1', 'evidence_2', 'evidence_1_stance', 'evidence_2_stance', 
                                'evidence_1_detection_score', 'evidence_2_detection_score']]

output_path = 'filtered_data.csv'
filtered_data.to_csv(output_path, index=False)
