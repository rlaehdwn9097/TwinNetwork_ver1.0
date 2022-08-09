import config as cf
import numpy as np

class Content(object):
    def __init__(self, _title, _size, _popularity, _peak_day,_category):
        self.title = _title
        self.size = _size
        self.popularity = _popularity
        self.peak_day =_peak_day
        self.category = _category
    
    def get_title(self):
        return self.title

    def get_popularity(self):
        return self.popularity

class contentStorage(object):
    def __init__(self, _size):
        self.capacity = _size
        self.stored = 0
        self.storage=[]
        self.lastdatelist=[]
        

    def abletostore(self,c:Content):
        freeSpace = self.capacity-self.stored
        if(freeSpace>=c.size):
            return 1
        else:
            return 0 
    def addContent(self,c:Content, date):
        self.storage.append(c)
        self.lastdatelist.append(date)
        self.stored = self.stored + c.size

    def isstored(self,c:Content):
        if len(self.storage)>0:
            for i in self.storage:
                if i.title == c.title:
                    return 1
        return 0

    def delContent(self,c:Content):
        newstorage=[]
        newlastdatelist=[]
        for i in range(len(self.storage)):
            if self.storage[i] is c:
                self.stored = self.stored-c.size
                #print(c.get_title(), "삭제 됌")
            else:
                newstorage.append(self.storage[i])
                newlastdatelist.append(self.lastdatelist[i])
        self.storage=newstorage
        self.lastdatelist=newlastdatelist

    def delFirstStored(self):#사용한지 가장 오래된 매체 삭제
        self.stored = self.stored - self.storage[0].size 
        self.storage=self.storage[1:]
        self.lastdatelist=self.lastdatelist[1:]
    


def updatequeue(path:list,c:Content,microBSList,BSList,dataCenter, date):
    if len(path) == 2:
        microBSList[path[1]].storage.delContent(c)
        microBSList[path[1]].storage.addContent(c,date)
        #print("MICROBS UPDATE")
        #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))
    if len(path) == 3:
        BSList[path[2]].storage.delContent(c)
        BSList[path[2]].storage.addContent(c,date)
        #print("BS UPDATE")
        #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))
    if len(path) == 4:
        dataCenter.storage.delContent(c)
        dataCenter.storage.addContent(c,date)
        #print("DATACENTER UPDATE")
        #print("updated_content : {} ==> current_date : {}".format(c.__dict__,date))


#def count_redundancy(contentList:list,self):
