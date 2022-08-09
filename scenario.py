from doctest import testsource
import content as ct
import numpy as np
import random
class Scenario(object):
    def __init__(self,_contentList:list):
        self.contentList=_contentList
        self.titleList = []
        self.set_titleList()

    
    def requestGenerate(self,_day):
        titleList = []
        weightList = []
        for i in range(len(self.contentList)):
            titleList.append(self.contentList[i].title)
            howAfter = abs(self.contentList[i].peak_day-_day)
            if howAfter>4:
                howAfter=7-howAfter
            weight = round(gaussian(0,howAfter,2)*self.contentList[i].popularity,4)
            weightList.append(weight)
        choice = random.choices(titleList, weights = weightList, k = 1)
        for i in self.contentList:
            if i.title == choice[0]:
                return i
    
    def set_titleList(self):
        for i in range(len(self.contentList)):
            self.titleList.append(self.contentList[i].title)

        # 중복제거
        self.titleList = set(self.titleList)
        self.titleList = list(self.titleList)
    
    def get_titleList(self):
        return self.titleList

            
def gaussian(x,mean,sigma):
    return(1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x-mean)**2/(2*sigma**2))
emBB =[

#일요일 ###
ct.Content('현재는 아름다워',20,25,0,"drama"),
ct.Content('미운 우리 새끼',20,11.8,0,"comedy"),
ct.Content('태종 이방원',20,11.5,0,"drama"),
ct.Content('1박 2일 시즌4',20,8.4,0,"comedy"),
ct.Content('KBS 뉴스 9',20,8.1,0,"news"),
ct.Content('TV 동물농장',20,7.6,0,"comedy"),
ct.Content('사장님 귀는 당나귀 귀',20,5.6,0,"comedy"),
ct.Content('복면가왕',20,5.6,0,"comedy"),
ct.Content('이슈 Pick, 쌤과 함께',20,5.1,0,"news"),
ct.Content('역사저널 그날',20,5.1,0,"news"),
ct.Content('코로나19 통합뉴스룸',20,5.1,0,"news"),
ct.Content('전국노래자랑',20,4.7,0,"comedy"),
ct.Content('노래가 좋아',20,3.9,0,"comedy"),
ct.Content('집사부 일체',20,3.9,0,"comedy"),
ct.Content('동물의 왕국',20,3.8,0,"news"),

ct.Content('Sunday_news_pop_8',20,8,0,"news"),
ct.Content('Sunday_news_pop_7.6',20,7.6,0,"news"),
ct.Content('Sunday_news1',20,1.2,0,"news"),
ct.Content('Sunday_news2',20,1.6,0,"news"),
ct.Content('Sunday_news3',20,1.1,0,"news"),
ct.Content('Sunday_news4',20,1.2,0,"news"),
ct.Content('Sunday_news5',20,1.6,0,"news"),
ct.Content('Sunday_news6',20,1.1,0,"news"),
ct.Content('Sunday_comedy1',20,1.3,0,"comedy"),
ct.Content('Sunday_comedy2',20,1.7,0,"comedy"),
ct.Content('Sunday_comedy3',20,1.4,0,"comedy"),
ct.Content('Sunday_comedy4',20,1.3,0,"comedy"),
ct.Content('Sunday_comedy5',20,1.7,0,"comedy"),
ct.Content('Sunday_comedy6',20,1.4,0,"comedy"),
ct.Content('Sunday_drama1',20,1.9,0,"drama"),
ct.Content('Sunday_drama2',20,1.6,0,"drama"),
ct.Content('Sunday_drama3',20,1.2,0,"drama"),
ct.Content('Sunday_drama4',20,1.9,0,"drama"),
ct.Content('Sunday_drama5',20,1.6,0,"drama"),
ct.Content('Sunday_drama6',20,1.2,0,"drama"),


###월요일###
ct.Content('으라차차 내 인생',20,17.1,1,"drama"),
ct.Content('사랑의 꽈배기',20,15.3,1,"drama"),
ct.Content('KBS 뉴스 9',20,10.7,1,"news"),
ct.Content('붉은 단심',20,6.3,1,"drama"),
ct.Content('비밀의 집',20,5.4,1,"drama"),
ct.Content('동상이몽 2',20,4.6,1,"comedy"),
ct.Content('생활의 달인',20,4.6,1,"comedy"),
ct.Content('코로나19 통합뉴스룸',20,10.7,1,"news"),
ct.Content('으라차차 내 인생',20,1.3,1,"drama"),
ct.Content('크레이지 러브',20,1.2,1,"drama"),

ct.Content('Monday_news_pop_8',20,8,0,"news"),
ct.Content('Monday_news_pop_7.6',20,7.6,0,"news"),
ct.Content('Monday_news1',20,1.2,1,"news"),
ct.Content('Monday_news2',20,1.6,1,"news"),
ct.Content('Monday_news3',20,1.1,1,"news"),
ct.Content('Monday_news4',20,1.2,1,"news"),
ct.Content('Monday_news5',20,1.6,1,"news"),
ct.Content('Monday_news6',20,1.1,1,"news"),
ct.Content('Monday_comedy1',20,1.3,1,"comedy"),
ct.Content('Monday_comedy2',20,1.7,1,"comedy"),
ct.Content('Monday_comedy3',20,1.4,1,"comedy"),
ct.Content('Monday_comedy4',20,1.3,1,"comedy"),
ct.Content('Monday_comedy5',20,1.7,1,"comedy"),
ct.Content('Monday_comedy6',20,1.4,1,"comedy"),
ct.Content('Monday_drama1',20,1.9,1,"drama"),
ct.Content('Monday_drama2',20,1.6,1,"drama"),
ct.Content('Monday_drama3',20,1.2,1,"drama"),
ct.Content('Monday_drama4',20,1.9,1,"drama"),
ct.Content('Monday_drama5',20,1.6,1,"drama"),
ct.Content('Monday_drama6',20,1.2,1,"drama"),


##화요일###
ct.Content('으라차차 내 인생',20,16.7,2,"drama"),
ct.Content('사랑의 꽈배기',20,16.3,2,"drama"),
ct.Content('KBS 뉴스 9',20,9.9,2,"drama"),
ct.Content('인간극장',20,9.5,2,"news"),
ct.Content('신발 벗고 돌싱포맨',20,6.4,2,"comedy"),
ct.Content('붉은 단심',20,6.0,2,"drama"),
ct.Content('순간포착 세상에 이런일이',20,5.7,2,"comedy"),
ct.Content('비밀의 집',20,5.3,2,"drama"),
ct.Content('코로나19 통합뉴스룸',20,4.5,2,"news"),

ct.Content('Tuesday_news_pop_8',20,8,2,"news"),
ct.Content('Tuesday_news_pop_7.6',20,7.6,2,"news"),
ct.Content('Tuesday_news1',20,1.2,2,"news"),
ct.Content('Tuesday_news2',20,1.6,2,"news"),
ct.Content('Tuesday_news3',20,1.1,2,"news"),
ct.Content('Tuesday_news4',20,1.2,2,"news"),
ct.Content('Tuesday_news5',20,1.6,2,"news"),
ct.Content('Tuesday_news6',20,1.1,2,"news"),
ct.Content('Tuesday_comedy1',20,1.3,2,"comedy"),
ct.Content('Tuesday_comedy2',20,1.7,2,"comedy"),
ct.Content('Tuesday_comedy3',20,1.4,2,"comedy"),
ct.Content('Tuesday_comedy4',20,1.3,2,"comedy"),
ct.Content('Tuesday_comedy5',20,1.7,2,"comedy"),
ct.Content('Tuesday_comedy6',20,1.4,2,"comedy"),
ct.Content('Tuesday_drama1',20,1.9,2,"drama"),
ct.Content('Tuesday_drama2',20,1.6,2,"drama"),
ct.Content('Tuesday_drama3',20,1.2,2,"drama"),
ct.Content('Tuesday_drama4',20,1.9,2,"drama"),
ct.Content('Tuesday_drama5',20,1.6,2,"drama"),
ct.Content('Tuesday_drama6',20,1.2,2,"drama"),

##수요일##
ct.Content('으라차차 내 인생',20,15.7,3,"drama"),
ct.Content('사랑의 꽈배기',20,14.1,3,"drama"),
ct.Content('KBS 뉴스 9',20,9.6,3,"news"),
ct.Content('인간극장',20,9.4,3,"news"),
ct.Content('골 때리는 그녀들',20,7.0,3,"comedy"),
ct.Content('라디오스타',20,5.9,3,"comedy"),
ct.Content('코로나19 통합뉴스룸',20,5.4,3,"news"),
ct.Content('비밀의 집',20,4.9,3,"drama"),
ct.Content('일꾼의 탄생',20,4.9,3,"drama"),
ct.Content('옥탑방의 문제아들',20,3.5,3,"comedy"),

ct.Content('Wednesday_news_pop_8',20,8,3,"news"),
ct.Content('Wednesday_news_pop_7.6',20,7.6,3,"news"),
ct.Content('Wednesday_news1',20,1.2,3,"news"),
ct.Content('Wednesday_news2',20,1.6,3,"news"),
ct.Content('Wednesday_news3',20,1.1,3,"news"),
ct.Content('Wednesday_news4',20,1.2,3,"news"),
ct.Content('Wednesday_news5',20,1.6,3,"news"),
ct.Content('Wednesday_news6',20,1.1,3,"news"),
ct.Content('Wednesday_comedy1',20,1.3,3,"comedy"),
ct.Content('Wednesday_comedy2',20,1.7,3,"comedy"),
ct.Content('Wednesday_comedy3',20,1.4,3,"comedy"),
ct.Content('Wednesday_comedy4',20,1.3,3,"comedy"),
ct.Content('Wednesday_comedy5',20,1.7,3,"comedy"),
ct.Content('Wednesday_comedy6',20,1.4,3,"comedy"),
ct.Content('Wednesday_drama1',20,1.9,3,"drama"),
ct.Content('Wednesday_drama2',20,1.6,3,"drama"),
ct.Content('Wednesday_drama3',20,1.2,3,"drama"),
ct.Content('Wednesday_drama4',20,1.9,3,"drama"),
ct.Content('Wednesday_drama5',20,1.6,3,"drama"),
ct.Content('Wednesday_drama6',20,1.2,3,"drama"),

##목요일##
ct.Content('으라차차 내 인생',20,16.2,4,"drama"),
ct.Content('사랑의 꽈배기',20,14.2,4,"drama"),
ct.Content('KBS 뉴스 9',20,10.2,4,"news"),
ct.Content('인간극장',20,8.5,4,"news"),
ct.Content('비밀의 집',20,8.8,4,"drama"),
ct.Content('코로나19 통합뉴스룸',20,4.8,4,"news"),
ct.Content('TV 동물농장',20,4.5,4,"comedy"),
ct.Content('걸어서 세계속으로',20,3.9,4,"news"),

ct.Content('Thursday_news_pop_8',20,8,4,"news"),
ct.Content('Thursday_news_pop_7.6',20,7.6,4,"news"),
ct.Content('Thursday_news1',20,1.2,4,"news"),
ct.Content('Thursday_news2',20,1.6,4,"news"),
ct.Content('Thursday_news3',20,1.1,4,"news"),
ct.Content('Thursday_news4',20,1.2,4,"news"),
ct.Content('Thursday_news5',20,1.6,4,"news"),
ct.Content('Thursday_news6',20,1.1,4,"news"),
ct.Content('Thursday_comedy1',20,1.3,4,"comedy"),
ct.Content('Thursday_comedy2',20,1.7,4,"comedy"),
ct.Content('Thursday_comedy3',20,1.4,4,"comedy"),
ct.Content('Thursday_comedy4',20,1.3,4,"comedy"),
ct.Content('Thursday_comedy5',20,1.7,4,"comedy"),
ct.Content('Thursday_comedy6',20,1.4,4,"comedy"),
ct.Content('Thursday_drama1',20,1.9,4,"drama"),
ct.Content('Thursday_drama2',20,1.6,4,"drama"),
ct.Content('Thursday_drama3',20,1.2,4,"drama"),
ct.Content('Thursday_drama4',20,1.9,4,"drama"),
ct.Content('Thursday_drama5',20,1.6,4,"drama"),
ct.Content('Thursday_drama6',20,1.2,4,"drama"),

##금요일##
ct.Content('으라차차 내 인생',20,14.7,5,"drama"),
ct.Content('사랑의 꽈배기',20,13.3,5,"drama"),
ct.Content('어게인 마이 라이프',20,10.1,5,"drama"),
ct.Content('인간극장',20,9.6,5,"news"),
ct.Content('KBS 뉴스 9',20,9.4,5,"news"),
ct.Content('나 혼자 산다',20,6.5,5,"comedy"),
ct.Content('비밀의 집',20,5.0,5,"drama"),
ct.Content('코로나19 통합뉴스룸',20,4.8,5,"news"),
ct.Content('자연의 철학자들',20,4.5,5,"news"),

ct.Content('Friday_news_pop_8',20,8,5,"news"),
ct.Content('Friday_news_pop_7.6',20,7.6,5,"news"),
ct.Content('Friday_news1',20,1.2,5,"news"),
ct.Content('Friday_news2',20,1.6,5,"news"),
ct.Content('Friday_news3',20,1.1,5,"news"),
ct.Content('Friday_news4',20,1.2,5,"news"),
ct.Content('Friday_news5',20,1.6,5,"news"),
ct.Content('Friday_news6',20,1.1,5,"news"),
ct.Content('Friday_comedy1',20,1.3,5,"comedy"),
ct.Content('Friday_comedy2',20,1.7,5,"comedy"),
ct.Content('Friday_comedy3',20,1.4,5,"comedy"),
ct.Content('Friday_comedy4',20,1.3,5,"comedy"),
ct.Content('Friday_comedy5',20,1.7,5,"comedy"),
ct.Content('Friday_comedy6',20,1.4,5,"comedy"),
ct.Content('Friday_drama1',20,1.9,5,"drama"),
ct.Content('Friday_drama2',20,1.6,5,"drama"),
ct.Content('Friday_drama3',20,1.2,5,"drama"),
ct.Content('Friday_drama4',20,1.9,5,"drama"),
ct.Content('Friday_drama5',20,1.6,5,"drama"),
ct.Content('Friday_drama6',20,1.2,5,"drama"),

##토요일##
ct.Content('현재는 아름다워',20,20.7,6,"drama"),
ct.Content('어게인 마이 라이프',20,9.6,6,"drama"),
ct.Content('KBS 뉴스 9',20,8.1,6,"news"),
ct.Content('놀면 뭐하니?',20,6.3,6,"comedy"),
ct.Content('불후의 명곡',20,5.7,6,"drama"),
ct.Content('살림하는 남자들 시즌2',20,5.6,6,"comedy"),
ct.Content('시니어 토크쇼 황금연못',20,5.0,6,"comedy"),
ct.Content('코로나19 통합뉴스룸',20,5.0,6,"news"),
ct.Content('전지적 참견 시점',20,4.4,6,"comedy"),

ct.Content('Saturday_news_pop_8',20,8,6,"news"),
ct.Content('Saturday_news_pop_7.6',20,7.6,6,"news"),
ct.Content('Saturday_news1',20,1.2,6,"news"),
ct.Content('Saturday_news2',20,1.6,6,"news"),
ct.Content('Saturday_news3',20,1.1,6,"news"),
ct.Content('Saturday_news4',20,1.2,6,"news"),
ct.Content('Saturday_news5',20,1.6,6,"news"),
ct.Content('Saturday_news6',20,1.1,6,"news"),
ct.Content('Saturday_comedy1',20,1.3,6,"comedy"),
ct.Content('Saturday_comedy2',20,1.7,6,"comedy"),
ct.Content('Saturday_comedy3',20,1.4,6,"comedy"),
ct.Content('Saturday_comedy4',20,1.3,6,"comedy"),
ct.Content('Saturday_comedy5',20,1.7,6,"comedy"),
ct.Content('Saturday_comedy6',20,1.4,6,"comedy"),
ct.Content('Saturday_drama1',20,1.9,6,"drama"),
ct.Content('Saturday_drama2',20,1.6,6,"drama"),
ct.Content('Saturday_drama3',20,1.2,6,"drama"),
ct.Content('Saturday_drama4',20,1.9,6,"drama"),
ct.Content('Saturday_drama5',20,1.6,6,"drama"),
ct.Content('Saturday_drama6',20,1.2,6,"drama"),

]

emBBScenario = Scenario(emBB)

contentLabel = {
'Saturday_drama4': 1, 'Monday_news_pop_7.6': 2, '인간극장': 3, 'Friday_comedy2': 4, 
'Saturday_news6': 5, 'KBS 뉴스 9': 6, 'Sunday_drama1': 7, 'Monday_comedy5': 8, 
'Tuesday_comedy3': 9, '코로나19 통합뉴스룸': 10, 'Sunday_news5': 11, '사장님 귀는 당나귀 귀': 12, 
'Friday_comedy3': 13, 'TV 동물농장': 14, 'Tuesday_news1': 15, 'Sunday_drama3': 16, 
'신발 벗고 돌싱포맨': 17, 'Saturday_comedy5': 18, 'Sunday_comedy1': 19, 'Wednesday_drama6': 20, 
'동물의 왕국': 21, 'Sunday_comedy5': 22, '라디오스타': 23, 'Monday_drama1': 24, 
'Friday_news2': 25, 'Saturday_comedy2': 26, '나 혼자 산다': 27, '역사저널 그날': 28, 
'Friday_drama2': 29, 'Sunday_drama5': 30, 'Tuesday_comedy1': 31, 'Thursday_comedy4': 32, 
'Friday_news_pop_8': 33, 'Saturday_drama6': 34, 'Tuesday_drama6': 35, 'Tuesday_drama2': 36, 
'Thursday_drama4': 37, 'Thursday_news_pop_7.6': 38, 'Friday_drama5': 39, 'Saturday_drama3': 40, 
'Saturday_comedy3': 41, '시니어 토크쇼 황금연못': 42, 'Thursday_drama1': 43, '동상이몽 2': 44, 
'Monday_drama5': 45, '순간포착 세상에 이런일이': 46, 'Saturday_comedy1': 47, 'Tuesday_news6': 48, 
'Sunday_news2': 49, 'Saturday_news1': 50, 'Sunday_news_pop_7.6': 51, 'Wednesday_comedy2': 52, 
'현재는 아름다워': 53, 'Sunday_news6': 54, 'Tuesday_news5': 55, 'Wednesday_drama2': 56, 
'태종 이방원': 57, 'Wednesday_drama1': 58, '골 때리는 그녀들': 59, 'Wednesday_comedy5': 60, 
'Thursday_drama5': 61, '전국노래자랑': 62, 'Thursday_comedy3': 63, '놀면 뭐하니?': 64, 
'Tuesday_comedy6': 65, 'Saturday_news4': 66, 'Thursday_news4': 67, 'Saturday_comedy4': 68, 
'Thursday_news2': 69, 'Friday_news4': 70, 'Monday_comedy4': 71, 'Monday_comedy2': 72, 
'붉은 단심': 73, 'Tuesday_news4': 74, 'Wednesday_news1': 75, 'Thursday_drama2': 76, 
'Friday_comedy5': 77, 'Monday_drama6': 78, '불후의 명곡': 79, 'Wednesday_news_pop_7.6': 80, 
'Friday_news6': 81, 'Monday_comedy1': 82, 'Monday_drama2': 83, 'Sunday_news_pop_8': 84, 
'Thursday_comedy6': 85, 'Saturday_comedy6': 86, 'Sunday_comedy6': 87, '이슈 Pick, 쌤과 함께': 88, 
'Thursday_comedy1': 89, 'Tuesday_news2': 90, '살림하는 남자들 시즌2': 91, 'Wednesday_drama5': 92, 
'1박 2일 시즌4': 93, '집사부 일체': 94, 'Wednesday_drama4': 95, 'Friday_drama4': 96, 
'Wednesday_drama3': 97, 'Wednesday_news_pop_8': 98, 'Tuesday_comedy4': 99, 'Sunday_drama6': 100, 
'Wednesday_comedy6': 101, '미운 우리 새끼': 102, '생활의 달인': 103, 'Monday_news2': 104, 
'Monday_drama4': 105, 'Friday_drama3': 106, 'Friday_news1': 107, 'Tuesday_drama5': 108, 
'Sunday_comedy4': 109, 'Wednesday_news3': 110, 'Wednesday_news2': 111, '비밀의 집': 112, 
'일꾼의 탄생': 113, 'Friday_drama1': 114, 'Wednesday_news4': 115, 'Thursday_news_pop_8': 116, 
'Sunday_drama4': 117, 'Sunday_news1': 118, 'Monday_news4': 119, 'Saturday_drama1': 120, 
'노래가 좋아': 121, 'Tuesday_drama4': 122, 'Saturday_news3': 123, 'Monday_drama3': 124, 
'Tuesday_comedy2': 125, '걸어서 세계속으로': 126, '복면가왕': 127, 'Thursday_comedy5': 128, 
'Monday_news6': 129, '전지적 참견 시점': 130, '크레이지 러브': 131, 'Thursday_drama6': 132, 
'Tuesday_drama1': 133, 'Saturday_news_pop_7.6': 134, 'Monday_comedy6': 135, 'Sunday_news4': 136, 
'Friday_news3': 137, 'Tuesday_news_pop_8': 138, 'Wednesday_news5': 139, 'Sunday_drama2': 140, 
'사랑의 꽈배기': 141, 'Monday_news5': 142, 'Sunday_comedy2': 143, 'Friday_news_pop_7.6': 144,
'으라차차 내 인생': 145, 'Monday_news_pop_8': 146, 'Sunday_news3': 147, 'Wednesday_news6': 148, 
'Saturday_news_pop_8': 149, 'Tuesday_comedy5': 150, 'Tuesday_drama3': 151, 'Monday_comedy3': 152, 
'Saturday_news5': 153, 'Sunday_comedy3': 154, 'Wednesday_comedy3': 155, 'Monday_news3': 156, 
'Tuesday_news_pop_7.6': 157, 'Wednesday_comedy4': 158, 'Friday_news5': 159, 'Thursday_news3': 160, 
'Thursday_news5': 161, 'Thursday_drama3': 162, 'Friday_comedy1': 163, 'Thursday_news6': 164, 
'Wednesday_comedy1': 165, '자연의 철학자들': 166, 'Tuesday_news3': 167, 'Friday_drama6': 168, 
'Friday_comedy6': 169, 'Friday_comedy4': 170, 'Saturday_drama2': 171, 'Monday_news1': 172, 
'어게인 마이 라이프': 173, 'Saturday_news2': 174, 'Thursday_comedy2': 175, 'Saturday_drama5': 176, 
'Thursday_news1': 177, '옥탑방의 문제아들': 178
}

#testScenario = [ct.Content('a',20),ct.Content('b',20),ct.Content('c',20),ct.Content('d',20),ct.Content('e',20),ct.Content('f',20),ct.Content('g',20),ct.Content('h',20),ct.Content('i',20),ct.Content('j',20)]

testScenario = [ct.Content('a',20,25,1,"drama")]