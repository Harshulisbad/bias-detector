import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
import torch
import requests
import sys
import re
import numpy as np

API_KEY = "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"
head = {"Authorization": f"Bearer {API_KEY}"}
API_URL = "https://api-inference.huggingface.co/models/hersheys-baklava/IsraelPalestine-Bias-Detector"

def query(payload):
    response = requests.post(API_URL, headers=head, json=payload)
    return response.json()

labels = ['-', "n", "+"] * 4
op = []
dis = True

def classification(text, out):
  Palestine = 0
  Israel=0
  PalestineM=0
  IsraelM=0
  classified = query({"inputs": text})
  scores = [0]*12
  
  for label in classified[0]:
    if label['label'] == "PS0":
      scores[0] = label['score']
    elif label['label'] == "PS1":
      scores[1] = label['score']
    elif label['label'] == "PS2":
      scores[2] = label['score']
    elif label['label'] == "IS0":
      scores[3] = label['score']
    elif label['label'] == "IS1":
      scores[4] = label['score']
    elif label['label'] == "IS2":
      scores[5] = label['score']
    elif label['label'] == "PM0":
      scores[6] = label['score']
    elif label['label'] == "PM1":
      scores[7] = label['score']
    elif label['label'] == "PM2":
      scores[8] = label['score']
    elif label['label'] == "IM0":
      scores[9] = label['score']
    elif label['label'] == "IM1":
      scores[10] = label['score']
    elif label['label'] == "IM2":
      scores[11] = label['score']

  scores = np.array(scores)
  PS = np.argmax(scores[0:3])
  IS = np.argmax(scores[3:6])
  PM = np.argmax(scores[6:9])
  IM = np.argmax(scores[9:12])

  if labels[PS] == "+":
    Palestine+=1
  elif labels[PS] == "-":
    Palestine-=1

  if labels[IS]=="+":
    Israel+=1
  elif labels[IS]=="-":
    Israel-=1

  if labels[PM]=="+":
    PalestineM+=1
  elif labels[PM]=="-":
    PalestineM-=1

  if labels[IM]=="+":
    IsraelM+=1
  elif labels[IM]=="-":
    IsraelM-=1
  list9=[]
  list9.append(labels[PS])
  list9.append(labels[IS])
  list9.append(labels[PM])
  list9.append(labels[IM])
  list9.append(text)

  if out:
    st.success((text +
                "  \nPS: "+ labels[PS] +
                "  \nIS: "+ labels[IS] +
                "  \nPM: "+labels[PM] +
                "  \nIM: "+labels[IM]))
  return list9


headers = {'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9"}

def predict(url, value):
  totalList=[]
  depths = {}
  page = requests.get(url, headers=headers)
  soup = BeautifulSoup(page.content,"html.parser")
  paragraphs = (soup.find_all("p"))

  for p in paragraphs:
    depth = 0
    for _ in p.parents:
      depth = depth + 1

    if str(depth) in depths:
      depths[str(depth)] = depths[str(depth)] + len(p.text)
    else:
      depths[str(depth)] = 1

  level = int(max(depths, key=lambda key: depths[key]))
  # while(depths[str(level)] > 100):
  #   del depths[str(level)]
  #   level = int(max(depths, key=lambda key: depths[key]))
  list1=[]
  list2=[]
  list3=[]
  list4=[]
  textCounter=0
  for p in paragraphs:
    d = 0
    for _ in p.parents:
      d = d + 1

    if (d == level):
      INP = p.get_text().strip()
      if INP != "":
        final=classification(INP, value)
        list1.append(final[0])
        list2.append(final[1])
        list3.append(final[2])
        list4.append(final[3])
        textCounter+=1
        totalList.append(final[4])

  if not value:
    outputs = [0, 0, 0, 0]
    list_of_lists = [list1, list2, list3, list4]
    for i in range(len(list_of_lists)):
        counter = 0
        for x in list_of_lists[i]:
            if x=="+":
                counter+=1
            elif x=="-":
                counter+=-1
        outputs[i] = (counter/textCounter)

    global op
    op = outputs


  #"""st.success((" PS: "+str(Palestine)+
              #"\nIS: "+str(Israel)+
              #"\nPM: "+str(PalestineM)+
              #"\nIM: "+str(IsraelM)))"""

def output():
    st.success("PS: "+str(op[0]))
    st.success("IS: "+str(op[1]))
    st.success("PM: "+str(op[2]))
    st.success("IM: "+str(op[3]))

def main():
  st.image("https://www.economist.com/cdn-cgi/image/width=1424,quality=80,format=auto/content-assets/images/20231021_CUP502.jpg")
  html_temp="""
  <style>
button {
    height: auto;
}
</style>
  <div style="background-color:#025246 ;padding:10px">
  <h2 style="color:white;text-align:center;">Israeli-Palestine Conflict Bias Detector</h2>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)


  with st.form('chat_input_form'):
    # Create two columns; adjust the ratio to your liking
    col1, col2 = st.columns([8,1])
    # Use the first column for text input
    with col1:
        link=st.text_input(".",placeholder="Enter a link", label_visibility="collapsed")
    # Use the second column for the submit button
    with col2:
        enter = st.form_submit_button("Enter")
        if enter:
          predict(link, False)
          with col1:
            output()
            global dis
            dis = False

  if st.button("More Details", disabled=dis):
      predict(link, True)

if __name__ == "__main__":
  main()
