# 字典类，继承于dict
class adict(dict):
    # 构造函数，之后传入了key:VOCAB value:{}
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        # 是用来存储对象属性的一个字典，其键为属性名，值为属性的值。
        self.__dict__ = self