mkdir -p ../data/xorqa/en/

curl https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_train_retrieve_eng_span.jsonl -o ../data/xorqa/en/xor_train_retrieve_eng_span.jsonl

curl https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1_1.jsonl -o ../data/xorqa/en/xor_dev_retrieve_eng_span_v1_1.jsonl

curl https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_test_retrieve_eng_span_q_only_v1_1.jsonl -o ../data/xorqa/en/xor_test_retrieve_eng_span_q_only_v1_1.jsonl