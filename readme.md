#### Make some change of rules.
Right now, the input is json file, since we will do regular expression matching later, I wrote every rule as follows:
```
  "rules":{
      "include":[".*(blog.jobbole.com).*",".*(blog.jobbole.com/\\d+).*"],
      "exclude": [".*(all-posts).*"],
      "textextraction":["(.*?Java.*)", "(.*?Python.*)"]
  }
```
