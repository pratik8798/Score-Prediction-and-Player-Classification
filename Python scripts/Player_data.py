# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:05:03 2019

@author: Dell
"""


from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import os

team_dict={"1":"England",
           "2":"Australia",
           "3":"South Africa",
           "4":"West Indies",
           "5":"New Zealand",
           "6":"India",
           "7":"Pakistan",
           "8":"Sri Lanka",
           "9":"Zimbabwe",
           "15":"Netherlands",
           "19":"Hong Kong",
           "20":"Papa New Guinea",
           "25":"Bangladesh",
           "27":"United Arab Emirates",
           "29":"Ireland",
           "30":"Scotland",
           "40":"Afghanistan"
           }

base='F:\STC\FINAL IMPLEMENTATION\DATA\CSV'


for ind,team in team_dict.items():
    
    
    file = os.path.join(base, team+"_batting.csv")         
    f=open(file,'w')
        
    headers="Player,Span,Matches,Innings,NO,Runs,Average,Strike Rate,100,50\n"
    f.write(headers)
    
    
    my_url='http://stats.espncricinfo.com/ci/engine/records/averages/batting.html?class=2;id='+ind+';type=team'
    #Opening connectiongrabbing the page and closing
        
    uClient=uReq(my_url)
    page_html=uClient.read()
    uClient.close()
        
        
    #parsing
        
    page_soup=soup(page_html,"html.parser")
        
    #grab each result
    table=page_soup.findAll("tr",{"class":"data1"})
    links=page_soup.findAll("a",{"class":"data-link"})
    for row in table:
        info=row.findAll("td",{"nowrap":"nowrap"})
        
        l=[]
        
        arr=info[1].text.split("-")
        if len(arr)!=0:
            if(int(arr[1])>=2010):
                
                #l.append(arr[1])
                for detail in info:
                    l.append(detail.text)
                    i+=1
            
            
                f.write(l[0]+","+l[1]+","+l[2]+","+l[3]+","+l[4]+","+l[5]+","+l[7]+","+l[9]+","+l[10]+","+l[11]+"\n")
            
            #print(l)
    
    
    f.close()



for ind,team in team_dict.items():
    
    file = os.path.join(base, team+"_bowling.csv")         
    f=open(file,'w')
        
    headers="Player,Span,Matches,Innings,Wickets,Best figures,Average,Economy Rate,Strike Rate,5Wkt Haul\n"
    f.write(headers)
    
    
    my_url='http://stats.espncricinfo.com/ci/engine/records/averages/bowling.html?class=2;id='+ind+';type=team'
    #Opening connectiongrabbing the page and closing
        
    uClient=uReq(my_url)
    page_html=uClient.read()
    uClient.close()
        
        
    #parsing
        
    page_soup=soup(page_html,"html.parser")
        
    #grab each result
    table=page_soup.findAll("tr",{"class":"data1"})
    links=page_soup.findAll("a",{"class":"data-link"})
    for row in table:
        info=row.findAll("td",{"nowrap":"nowrap"})
        
        l=[]
        
        arr=info[1].text.split("-")
        if len(arr)!=0:
            if(int(arr[1])>=2010):
                
                #l.append(arr[1])
                for detail in info:
                    l.append(detail.text)
                    i+=1
            
            
                f.write(l[0]+","+l[1]+","+l[2]+","+l[3]+","+l[6]+","+l[7]+","+l[8]+","+l[9]+","+l[10]+","+l[12]+"\n")
            
            #print(l)
    
    
    f.close()



