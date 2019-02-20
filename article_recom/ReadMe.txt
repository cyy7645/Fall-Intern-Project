recommendation.ipynb： includes all methods and the final two APIs func1 and func2.

func1：call it every hour, it's going for update TD-IDF table of dataset, and some temporary variables.

func2：call it in real-time for every recommendation. Write the most N similar documents' ids into write_keys.txt.

chinese_stop_words.txt： stop-words of Chinese.

new_doc.txt： test list of articles wating for being recommended. 

news： original dataset.

tmp： temporary files, the return data of func1.

write_keys.txt： the results will be in write_keys.txt (ids of articles)
