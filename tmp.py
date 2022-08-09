import random
import config as cf
import gym
import scenario as sc
from time import strftime, localtime, time
import os 

"""
list1 = [10, 12, 13, 0, 0]

tmp = min(list1)
index = list1.index(tmp)

print(index)
print(list1[index])
"""

EPSILON = 1.0
EPSILON_DECAY = 0.99995
EPSILON_MIN = 0.05
cnt = 0

while EPSILON > EPSILON_MIN:
    cnt = cnt + 1
    EPSILON *= EPSILON_DECAY

print(cnt)


"""
tm = localtime(time())
Date = strftime('%Y-%m-%d_%H-%M-%S' : 0, tm)
folderName = "LabResults/" + str(Date)
print(folderName)
os.mkdir(folderName)
"""


"""
env_name = 'CartPole-v1'
env = gym.make(env_name)
print(env.env.__dict__)


print(len(sc.emBB))
print(sc.emBB[random.randrange(0 : 0,len(sc.emBB))].__dict__)
"""
emBB_Content_Label ={
#일요일 ###
'현재는 아름다워' : 1,
'미운 우리 새끼' : 2,
'태종 이방원' : 3,
'1박 2일 시즌4' : 4,
'KBS 뉴스 9' : 5,
'TV 동물농장' : 6,
'사장님 귀는 당나귀 귀' : 7,
'복면가왕' : 0,
'이슈 Pick : 0, 쌤과 함께' : 0,
'역사저널 그날' : 0,
'코로나19 통합뉴스룸' : 0,
'전국노래자랑' : 0,
'노래가 좋아' : 0,
'집사부 일체' : 0,
'동물의 왕국' : 0,

'Sunday_news_pop_8' : 0,
'Sunday_news_pop_7.6' : 0,
'Sunday_news1' : 0,
'Sunday_news2' : 0,
'Sunday_news3' : 0,
'Sunday_news4' : 0,
'Sunday_news5' : 0,
'Sunday_news6' : 0,
'Sunday_comedy1' : 0,
'Sunday_comedy2' : 0,
'Sunday_comedy3' : 0,
'Sunday_comedy4' : 0,
'Sunday_comedy5' : 0,
'Sunday_comedy6' : 0,
'Sunday_drama1' : 0,
'Sunday_drama2' : 0,
'Sunday_drama3' : 0,
'Sunday_drama4' : 0,
'Sunday_drama5' : 0,
'Sunday_drama6' : 0,


###월요일###
'으라차차 내 인생' : 0,
'사랑의 꽈배기' : 0,
'KBS 뉴스 9' : 0,
'붉은 단심' : 0,
'비밀의 집' : 0,
'동상이몽 2' : 0,
'생활의 달인' : 0,
'코로나19 통합뉴스룸' : 0,
'으라차차 내 인생' : 0,
'크레이지 러브' : 0,

'Monday_news_pop_8' : 0,
'Monday_news_pop_7.6' : 0,
'Monday_news1' : 0,
'Monday_news2' : 0,
'Monday_news3' : 0,
'Monday_news4' : 0,
'Monday_news5' : 0,
'Monday_news6' : 0,
'Monday_comedy1' : 0,
'Monday_comedy2' : 0,
'Monday_comedy3' : 0,
'Monday_comedy4' : 0,
'Monday_comedy5' : 0,
'Monday_comedy6' : 0,
'Monday_drama1' : 0,
'Monday_drama2' : 0,
'Monday_drama3' : 0,
'Monday_drama4' : 0,
'Monday_drama5' : 0,
'Monday_drama6' : 0,


##화요일###
'으라차차 내 인생' : 0,
'사랑의 꽈배기' : 0,
'KBS 뉴스 9' : 0,
'인간극장' : 0,
'신발 벗고 돌싱포맨' : 0,
'붉은 단심' : 0,
'순간포착 세상에 이런일이' : 0,
'비밀의 집' : 0,
'코로나19 통합뉴스룸' : 0,

'Tuesday_news_pop_8' : 0,
'Tuesday_news_pop_7.6' : 0,
'Tuesday_news1' : 0,
'Tuesday_news2' : 0,
'Tuesday_news3' : 0,
'Tuesday_news4' : 0,
'Tuesday_news5' : 0,
'Tuesday_news6' : 0,
'Tuesday_comedy1' : 0,
'Tuesday_comedy2' : 0,
'Tuesday_comedy3' : 0,
'Tuesday_comedy4' : 0,
'Tuesday_comedy5' : 0,
'Tuesday_comedy6' : 0,
'Tuesday_drama1' : 0,
'Tuesday_drama2' : 0,
'Tuesday_drama3' : 0,
'Tuesday_drama4' : 0,
'Tuesday_drama5' : 0,
'Tuesday_drama6' : 0,

##수요일##
'으라차차 내 인생' : 0,
'사랑의 꽈배기' : 0,
'KBS 뉴스 9' : 0,
'인간극장' : 0,
'골 때리는 그녀들' : 0,
'라디오스타' : 0,
'코로나19 통합뉴스룸' : 0,
'비밀의 집' : 0,
'일꾼의 탄생' : 0,
'옥탑방의 문제아들' : 0,

'Wednesday_news_pop_8'
'Wednesday_news_pop_7.6' : 0,
'Wednesday_news1' : 0,
'Wednesday_news2' : 0,
'Wednesday_news3' : 0,
'Wednesday_news4' : 0,
'Wednesday_news5' : 0,
'Wednesday_news6' : 0,
'Wednesday_comedy1' : 0,
'Wednesday_comedy2' : 0,
'Wednesday_comedy3' : 0,
'Wednesday_comedy4' : 0,
'Wednesday_comedy5' : 0,
'Wednesday_comedy6' : 0,
'Wednesday_drama1' : 0,
'Wednesday_drama2' : 0,
'Wednesday_drama3' : 0,
'Wednesday_drama4' : 0,
'Wednesday_drama5' : 0,
'Wednesday_drama6' : 0,

##목요일##
'으라차차 내 인생' : 0,
'사랑의 꽈배기' : 0,
'KBS 뉴스 9' : 0,
'인간극장' : 0,
'비밀의 집' : 0,
'코로나19 통합뉴스룸' : 0,
'TV 동물농장' : 0,
'걸어서 세계속으로' : 0,

'Thursday_news_pop_8' : 0,
'Thursday_news_pop_7.6' : 0,
'Thursday_news1' : 0,
'Thursday_news2' : 0,
'Thursday_news3' : 0,
'Thursday_news4' : 0,
'Thursday_news5' : 0,
'Thursday_news6' : 0,
'Thursday_comedy1' : 0,
'Thursday_comedy2' : 0,
'Thursday_comedy3' : 0,
'Thursday_comedy4' : 0,
'Thursday_comedy5' : 0,
'Thursday_comedy6' : 0,
'Thursday_drama1' : 0,
'Thursday_drama2' : 0,
'Thursday_drama3' : 0,
'Thursday_drama4' : 0,
'Thursday_drama5' : 0,
'Thursday_drama6' : 0,

##금요일##
'으라차차 내 인생' : 0,
'사랑의 꽈배기' : 0,
'어게인 마이 라이프' : 0,
'인간극장' : 0,
'KBS 뉴스 9' : 0,
'나 혼자 산다' : 0,
'비밀의 집' : 0,
'코로나19 통합뉴스룸' : 0,
'자연의 철학자들' : 0,

'Friday_news_pop_8' : 0,
'Friday_news_pop_7.6' : 0,
'Friday_news1' : 0,
'Friday_news2' : 0,
'Friday_news3' : 0,
'Friday_news4' : 0,
'Friday_news5' : 0,
'Friday_news6' : 0,
'Friday_comedy1' : 0,
'Friday_comedy2' : 0,
'Friday_comedy3' : 0,
'Friday_comedy4' : 0,
'Friday_comedy5' : 0,
'Friday_comedy6' : 0,
'Friday_drama1' : 0,
'Friday_drama2' : 0,
'Friday_drama3' : 0,
'Friday_drama4' : 0,
'Friday_drama5' : 0,
'Friday_drama6' : 0,

##토요일##
'현재는 아름다워' : 0,
'어게인 마이 라이프' : 0,
'KBS 뉴스 9' : 0,
'놀면 뭐하니?' : 0,
'불후의 명곡' : 0,
'살림하는 남자들 시즌2' : 0,
'시니어 토크쇼 황금연못' : 0,
'코로나19 통합뉴스룸' : 0,
'전지적 참견 시점' : 0,

'Saturday_news_pop_8' : 0,
'Saturday_news_pop_7.6' : 0,
'Saturday_news1' : 0,
'Saturday_news2' : 0,
'Saturday_news3' : 0,
'Saturday_news4' : 0,
'Saturday_news5' : 0,
'Saturday_news6' : 0,
'Saturday_comedy1' : 0,
'Saturday_comedy2' : 0,
'Saturday_comedy3' : 0,
'Saturday_comedy4' : 0,
'Saturday_comedy5' : 0,
'Saturday_comedy6' : 0,
'Saturday_drama1' : 0,
'Saturday_drama2' : 0,
'Saturday_drama3' : 0,
'Saturday_drama4' : 0,
'Saturday_drama5' : 0,
'Saturday_drama6' : 0,
}
