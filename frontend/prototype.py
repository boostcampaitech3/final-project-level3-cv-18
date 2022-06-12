import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
import pandas as pd
import requests
from image_crop import crop
import datetime
from dateutil.tz import gettz
import time
from streamlit_option_menu import option_menu
from concurrent.futures import ThreadPoolExecutor


st.set_page_config(page_title = 'ARTLAB 피부평가',layout="wide")

def post_url(args):
    return requests.post(args[0],files=args[1])



import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def create_userinfo():
    c.execute('CREATE TABLE IF NOT EXISTS skin(username TEXT,date DATE, wrinkle INT, oil INT, sensitive INT, pigmentation INT, hydration INT)')
    conn.commit()


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def add_skin_analysis(username,date,wrinkle,oil,sensitive,pigmentation,hydration):
    c.execute('INSERT INTO skin(username,date,wrinkle,oil,sensitive,pigmentation,hydration) VALUES (?,?,?,?,?,?,?)',(username,date,wrinkle,oil,sensitive,pigmentation,hydration))
    conn.commit()

def view_skin_analysis(username):
    s = f"SELECT date,wrinkle,oil,sensitive,pigmentation,hydration FROM SKIN WHERE USERNAME='{username}'"
    c.execute(s)
    data = c.fetchall()
    return data

def check_user(username):
    s = f"SELECT * FROM userstable WHERE USERNAME='{username}'"
    c.execute(s)
    data = c.fetchone()
    return data

def image_upload():
    img_file = st.file_uploader('')
    if img_file : 
        image_bytes = img_file.getvalue()
        img =Image.open(io.BytesIO(image_bytes))
        encoded_img = np.frombuffer(img_file.getvalue(),dtype=np.uint8)
        new_img = cv2.imdecode(encoded_img,cv2.IMREAD_COLOR)
        try:
            crop(new_img)
            crop_img = cv2.cvtColor(crop(new_img),cv2.COLOR_BGR2RGB)
            files = [
                ('files', (img_file.name, image_bytes,
                        img_file.type))
            ]
            start = time.time()
            with st.spinner(text='In progress'):
                list_of_urls = [('http://101.101.217.13:30002/gradcam',files),('http://101.101.217.13:30004/gradcam',files),('http://115.85.183.56:30001/gradcam',files),('http://115.85.183.56:30002/gradcam',files),('http://115.85.183.56:30003/gradcam',files)]
                with ThreadPoolExecutor(max_workers=5) as pool:
                    response_list = list(pool.map(post_url,list_of_urls))
                wrinkle_cam_response = response_list[0] # 주름
                oil_cam_response = response_list[1] # 유분
                sensitive_cam_response = response_list[2] # 민감도
                pig_cam_response = response_list[3] # 색조
                hyd_cam_response = response_list[4] # 수분

                wrinkle_result = (np.array(wrinkle_cam_response.json()['cam'])) # 주름
                wrinkle_label = wrinkle_cam_response.json()['label'] # 주름

                oil_result = (np.array(oil_cam_response.json()['cam'])) # 유분
                oil_label = oil_cam_response.json()['label'] # 유분
                
    
                sensitive_result = (np.array(sensitive_cam_response.json()['cam']))
                sensitive_label = sensitive_cam_response.json()['label']

                pig_result = (np.array(pig_cam_response.json()['cam']))
                pig_label = pig_cam_response.json()['label']

                hyd_result = (np.array(hyd_cam_response.json()['cam']))
                hyd_label = hyd_cam_response.json()['label']

                col1,col2,col3,col4,col5 = st.columns(5)
                wrinkle,oil,sensitive,pigmentation,hydration=wrinkle_label,oil_label,sensitive_label,pig_label,hyd_label
                now = datetime.datetime.now(gettz('Asia/Seoul'))
                date = now.strftime("%Y-%m-%d-%H:%M")
                add_skin_analysis(username,date,wrinkle,oil,sensitive,pigmentation,hydration)
                c.close()
                with col1:
                    st.image(wrinkle_result,caption=f"주름 : {wrinkle}")
                with col2:
                    st.image(oil_result,caption=f"유분 : {oil}")
                with col3:
                    st.image(sensitive_result,caption=f"민감도 : {sensitive}")
                with col4:
                    st.image(pig_result,caption=f"색조 : {pigmentation}")
                with col5:
                    st.image(hyd_result,caption=f"수분 : {hydration}")
                end = time.time()
                st.write(end-start)
        except:
            st.warning('사진을 다시 찍어주세요.')

def camera_input():
    img_file = st.camera_input('')
    if img_file : 
        image_bytes = img_file.getvalue()
        img =Image.open(io.BytesIO(image_bytes))
        encoded_img = np.frombuffer(img_file.getvalue(),dtype=np.uint8)
        new_img = cv2.imdecode(encoded_img,cv2.IMREAD_COLOR)
        try:
            crop(new_img)
            crop_img = cv2.cvtColor(crop(new_img),cv2.COLOR_BGR2RGB)
            files = [
                ('files', (img_file.name, image_bytes,
                        img_file.type))
            ]
            start = time.time()
            with st.spinner(text='In progress'):
                list_of_urls = [('http://101.101.217.13:30002/gradcam',files),('http://101.101.217.13:30004/gradcam',files),('http://115.85.183.56:30001/gradcam',files),('http://115.85.183.56:30002/gradcam',files),('http://115.85.183.56:30003/gradcam',files)]
                with ThreadPoolExecutor(max_workers=5) as pool:
                    response_list = list(pool.map(post_url,list_of_urls))
                wrinkle_cam_response = response_list[0] # 주름
                oil_cam_response = response_list[1] # 유분
                sensitive_cam_response = response_list[2] # 민감도
                pig_cam_response = response_list[3] # 색조
                hyd_cam_response = response_list[4] # 수분


                wrinkle_result = (np.array(wrinkle_cam_response.json()['cam'])) # 주름
                wrinkle_label = wrinkle_cam_response.json()['label'] # 주름

                oil_result = (np.array(oil_cam_response.json()['cam'])) # 유분
                oil_label = oil_cam_response.json()['label'] # 유분
                

                sensitive_result = (np.array(sensitive_cam_response.json()['cam']))
                sensitive_label = sensitive_cam_response.json()['label']

                pig_result = (np.array(pig_cam_response.json()['cam']))
                pig_label = pig_cam_response.json()['label']

                hyd_result = (np.array(hyd_cam_response.json()['cam']))
                hyd_label = hyd_cam_response.json()['label']

                col1,col2,col3,col4,col5 = st.columns(5)
                wrinkle,oil,sensitive,pigmentation,hydration=wrinkle_label,oil_label,sensitive_label,pig_label,hyd_label
                now = datetime.datetime.now(gettz('Asia/Seoul'))
                date = now.strftime("%Y-%m-%d-%H:%M")
                add_skin_analysis(username,date,wrinkle,oil,sensitive,pigmentation,hydration)
                c.close()
                with col1:
                    st.image(wrinkle_result,caption=f"주름 : {wrinkle}")
                with col2:
                    st.image(oil_result,caption=f"유분 : {oil}")
                with col3:
                    st.image(sensitive_result,caption=f"민감도 : {sensitive}")
                with col4:
                    st.image(pig_result,caption=f"색조 : {pigmentation}")
                with col5:
                    st.image(hyd_result,caption=f"수분 : {hydration}")
                end = time.time()
                st.write(end-start)
        except:
            st.warning('사진을 다시 찍어주세요.')

def next_page():
    st.session_state.page+=1

def previous_page():
    st.session_state.page-=1

def user_update(name):
    st.session_state.username = name

def select_sigup():
    st.session_state.form = 'signup_form'



st.sidebar.image('/opt/ml/input/artlab/backend/logo/artlab.png')
with st.sidebar:
    choice = option_menu('Main menu',['Home','로그인','회원가입'],
    icons = ['box-arrow-in-left','gear'],menu_icon='menu-button',default_index=0)
    choice




if choice == "로그인":
    st.subheader("로그인 상태")

    username = st.text_input("User Name")
    password = st.text_input("Password",type='password')

    if 'username' not in st.session_state:
        st.session_state.username = ''
    user_update(username)

    if st.button("로그인"):
        create_usertable()
        create_userinfo()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))

        if result and st.session_state.username!='':

            st.sidebar.success("{}님 환영합니다.".format(username))
            logout = st.sidebar.button('로그아웃')
            if logout:
                user_update('')
                choice = 'Home'
                st.write(choice)
            st.subheader('수행하실 작업을 선택해주세요.')
            task = st.selectbox("",["분석","기록"])

            if task == "분석":
                st.subheader('피부 평가')
                with st.sidebar:
                    select_method = option_menu('분석방법',['사진 촬영','저장된 사진 업로드'],
                    icons = ['camera-fill','upload'],menu_icon='menu-button',default_index=0)
                    select_method
                if select_method == '사진 촬영':
                    st.markdown('### 사진을 촬영하여 피부를 검사하는 방법입니다.')
                    camera_input()
                else:
                    st.markdown('### 저장된사진을 통해 피부를 검사하는 방법입니다.')
                    st.markdown('#### 조금 더 정밀한 결과를 받아보실 수 있습니다.')
                    image_upload()

            elif task == "기록":
                st.subheader("평가 기록")
                st.markdown("## 사용자의 피부 진단 기록을 확인할 수 있는 공간입니다.")
                user_result = view_skin_analysis(username)
                c.close()
                clean_db = pd.DataFrame(user_result,columns=['날짜','주름','유분','민감도','색조','수분']).set_index('날짜')
                clean_db = clean_db.sort_values(by='날짜',ascending=False)
                maximum = len(clean_db)
                if 'page' not in st.session_state:
                    st.session_state.page = 0

                start_idx = 10*st.session_state.page
                end_idx = start_idx+10
                st.write("")
                st.write(clean_db.iloc[start_idx:end_idx])

                co1,co2,co3,_ = st.columns([0.1, 0.17, 0.1, 0.8])

                co2.write(f"Page of {1+st.session_state.page} of {round(maximum/10)+1}")

                if st.session_state.page < round(maximum/10):
                    co3.button(">",on_click=next_page)
                else:
                    co3.write("")
                
                if st.session_state.page > 0:
                    co1.button("<",on_click=previous_page)
                else:
                    co1.write("")
            
                
                with st.expander("그래프로 확인하기"):
                    clean_db = clean_db.iloc[start_idx:end_idx]
                    options = st.multiselect(
                    '검사 항목을 선택하세요.',
                    ['주름','유분','민감도','색조','수분'],
                    ['주름','유분','민감도','색조','수분']
                    )
                    clean_db = clean_db[options]
                    st.line_chart(clean_db)
        else:
            st.warning("아이디 또는 비밀번호가 맞지 않습니다.")

elif choice == "회원가입":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')

    if st.button("Signup"):
        create_usertable()
        check_valid = check_user(new_user)
        if check_valid:
            st.error("중복된 아이디 입니다.")
            c.close()
        else:
            add_userdata(new_user,make_hashes(new_password))
            c.close()
            st.success("계정 생성에 성공하셨습니다.")
            st.info("왼쪽 로그인 메뉴를 통해 로그인해주세요.")

elif choice == 'Home':
    st.image('/opt/ml/input/artlab/backend/logo/h_logo.png',use_column_width='always')
    st.title('ARTLAB 기업연계 프로젝트입니다.')
    st.markdown('### 사진을 찍거나 업로드하여 5개의 항목에 대해 평가하는 서비스입니다.')
    st.image('/opt/ml/input/artlab/backend/logo/example.png',use_column_width='always')
    st.markdown('### 사진을 찍거나 업로드하시면 각각의 항목에 대해 점수와 함께 grad-cam 결과를 얻습니다.')
    st.markdown('### 또한 저희 데모버전은 사용자 별 진단 기록을 저장하여 시각화하는 과정을 포함하고 있습니다.')




   

