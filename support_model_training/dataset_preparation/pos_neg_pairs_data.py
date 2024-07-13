import pandas as pd

file_path = 'filtered_data.csv'

data = pd.read_csv(file_path)

def remove_ref(sentence):
    if "[REF]" in sentence:
        return sentence.replace(" [REF]", "")
    return sentence

sorted_data = pd.DataFrame(columns=['topic', 'positive_evidence', 'negative_evidence_1', 'negative_evidence_2', 'negative_evidence_3', 
                                    'negative_evidence_4', 'negative_evidence_5'])

for idx, row in data.iterrows():
    # evidence labeled as "PRO"
    pro_evidence = row['evidence_1'] if row['evidence_1_stance'] == 'PRO' else row['evidence_2']
    # find all other rows containing the same pro evidence
    same_rows = data[(data['evidence_1'] == pro_evidence) | (data['evidence_2'] == pro_evidence)]

    # extract negative evidences with detection score
    negative_evidences = []
    for _, same_row in same_rows.iterrows():
        if same_row['evidence_1_stance'] == 'CON':
            negative_evidences.append((same_row['evidence_1'], same_row['evidence_1_detection_score']))
        if same_row['evidence_2_stance'] == 'CON':
            negative_evidences.append((same_row['evidence_2'], same_row['evidence_2_detection_score']))
    
    # sort negative evidences by detection score (from highest to lowest)
    negative_evidences.sort(key=lambda x: x[1], reverse=True)

    # top 5 negative evidences
    top_negative_evidences = [evi[0] for evi in negative_evidences[:5]]

    # Pad with the last available negative evidence if there are less than 5 negative evidences
    last_negative_evidence = top_negative_evidences[-1]
    while len(top_negative_evidences) < 5:
        top_negative_evidences.append(last_negative_evidence)
    
    # concatenate rows
    concatenated_row = {
        'topic': row['topic'],
        'positive_evidence': remove_ref(pro_evidence),
        'negative_evidence_1': remove_ref(top_negative_evidences[0]),
        'negative_evidence_2': remove_ref(top_negative_evidences[1]),
        'negative_evidence_3': remove_ref(top_negative_evidences[2]),
        'negative_evidence_4': remove_ref(top_negative_evidences[3]),
        'negative_evidence_5': remove_ref(top_negative_evidences[4])
    }

    sorted_data = pd.concat([sorted_data, pd.DataFrame([concatenated_row])], ignore_index=True)

sorted_data = sorted_data.drop_duplicates(subset=['positive_evidence']).reset_index(drop=True)

output_path = 'pos_neg_pairs_train.csv'
sorted_data.to_csv(output_path, index=False)


