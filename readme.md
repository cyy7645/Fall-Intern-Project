### Make some change of rules.
Right now, the input is json file, since we will do regular expression matching later, I wrote every rule as follows:
```
  "rules":{
      "include":[".*(blog.jobbole.com).*",".*(blog.jobbole.com/\\d+).*"],
      "exclude": [".*(all-posts).*"],
      "textextraction":["(.*?Java.*)", "(.*?Python.*)"]
  }
```
Like   
```
".*(blog.jobbole.com).*"
```
stands for that the url should contain "blog.jobbole.com".   

### What job does each file complete
There are four import files, they are <b>jobbole.py</b>, <b>items.py</b>, <b>pipelines.py</b>, <b>settings.py</b>   

- for <b>jobbole.py</b>:   
  it defines how to parse the pages (parse article list pages in <i>parse</i> method, parse article pages in <i>parse_detail</i> method), have a look at comments for more detailed info.
  
- for <b>items.py</b>:   
it defines the fields(like title, url, h1..) in table, and methods to process the form of items. Like convert the string to date, match exact numbers with regular expression. Finally, the items will be passed to <b>pipelines.py</b> automatically.  

- for <b>pipelines.py</b>:  
it defines the method of insertion query.

- for <b>settings.py</b>:  
there are two parts needed to be set.
1. call the methods of insertion query in <b>pipelines.py</b>. 
```
ITEM_PIPELINES = {
    # number stands for the priority
   'JobboleSpider.pipelines.MysqlTwistedPipline': 1,
}
```
2. setting for mysql, have to be changed as your dababase.  
```
# setting for mysql
MYSQL_HOST = "127.0.0.1"
MYSQL_DBNAME = "article_spider"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
PORT = 8889
```

### How to run
- create a table with following sql:
```
CREATE TABLE `jobbole_article` (
  `title` varchar(200) CHARACTER SET ucs2 NOT NULL DEFAULT '""',
  `create_date` date DEFAULT NULL,
  `url` varchar(300) CHARACTER SET ucs2 NOT NULL DEFAULT '""',
  `url_object_id` varchar(50) CHARACTER SET ucs2 NOT NULL DEFAULT 'hahah',
  `front_image_url` varchar(300) CHARACTER SET ucs2 DEFAULT '""',
  `comment_nums` int(11) DEFAULT '0',
  `fav_nums` int(11) DEFAULT '0',
  `praise_nums` int(11) DEFAULT '0',
  `tags` varchar(200) CHARACTER SET ucs2 DEFAULT '""',
  `content` longtext CHARACTER SET ucs2,
  `topics` varchar(200) DEFAULT NULL,
  `textExtraction` varchar(200) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

- change you setting of mysql. 
```
# setting for mysql
MYSQL_HOST = "127.0.0.1"
MYSQL_DBNAME = "article_spider"
MYSQL_USER = "root"
MYSQL_PASSWORD = "root"
PORT = 8889
```

- make some changes in input.json for different rules, make sure to write regular expression form.  
```
{
    "_comment": "regular expression in json is a little bit different, we use \\ to represent backslash",
    "urls": ["http://blog.jobbole.com/all-posts/"],
    "rules":{
        "include":[".*(blog.jobbole.com).*",".*(blog.jobbole.com/\\d+).*"],
        "exclude": [".*(all-posts).*"],
        "textextraction":["(.*?Java.*)", "(.*?Python.*)"]
    }
}
```

- run main.py after turn on mysql server.
