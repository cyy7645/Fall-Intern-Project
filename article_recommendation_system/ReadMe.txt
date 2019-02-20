Find the most N similar documents in dataset with given article.


dataset folder: includes many cvs files, every one is the collection of specific topic.
Attributes of every article: id, create_time, title, img_url, category, source, description, 
content, nick_name_uuid.

we only use：
id: the unique ID of document.(final results)

create_time: Creted time of document, we can search the similar documents within specified days.

title: title of document.
description: abstract of document.
content: content of document.

article_recommendation.ipynb: implemented the process in Python.
