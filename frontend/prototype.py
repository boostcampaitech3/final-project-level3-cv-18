import streamlit as st
from PIL import Image
import io
import numpy as np
import torch
import cv2
import pandas as pd
import requests
from backend.image_crop import crop


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
    c.execute('CREATE TABLE IF NOT EXISTS skin(username TEXT, wrinkle INT, oil INT, sensitive INT, pigmentation INT, hydration INT)')
    conn.commit()


def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def add_skin_analysis(username,wrinkle,oil,sensitive,pigmentation,hydration):
    c.execute('INSERT INTO skin(username,wrinkle,oil,sensitive,pigmentation,hydration) VALUES (?,?,?,?,?,?)',(username,wrinkle,oil,sensitive,pigmentation,hydration))
    conn.commit()

def view_skin_analysis(username):
    s = f"SELECT wrinkle,oil,sensitive,pigmentation,hydration FROM SKIN WHERE USERNAME='{username}'"
    c.execute(s)
    data = c.fetchall()
    return data

def check_user(username):
    s = f"SELECT * FROM userstable WHERE USERNAME='{username}'"
    c.execute(s)
    data = c.fetchone()
    return data

st.title("XAI 피부평가 Demo버전")

global crop_img
menu = ["Login","SignUp"]
choice = st.sidebar.selectbox("Menu",menu)


if choice == "Login":
    st.subheader("Login Section")

    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    if st.sidebar.checkbox("Login"):
        # if password == '12345':
        create_usertable()
        create_userinfo()
        hashed_pswd = make_hashes(password)

        result = login_user(username,check_hashes(password,hashed_pswd))
        if result:

            st.success("Logged In as {}".format(username))

            task = st.selectbox("Task",["Analytics","Profiles"])

            if task == "Analytics":
                st.subheader('피부 평가')
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
                        cam_response = requests.post("http://101.101.217.13:30002/gradcam", files=files)
                        cam_result = (np.array(cam_response.json()['cam']))
                        label = cam_response.json()['label']
                        col1,col2,col3,col4,col5 = st.columns(5)
                        wrinkle,oil,sensitive,pigmentation,hydration=label[0],label[1],label[2],label[3],label[4]
                        add_skin_analysis(username,wrinkle,oil,sensitive,pigmentation,hydration)
                        c.close()
                        with col1:
                            st.image(cam_result[0],caption=f"label(wrinkle) : {wrinkle}")
                        with col2:
                            st.image(cam_result[1],caption=f"label(oil) : {oil}")
                        with col3:
                            st.image(cam_result[2],caption=f"label(sensitive) : {sensitive}")
                        with col4:
                            st.image(cam_result[3],caption=f"label(pigmentation) : {pigmentation}")
                        with col5:
                            st.image(cam_result[4],caption=f"label(hydration) : {hydration}")
                    except:
                        st.warning('사진을 다시 찍어주세요.')
            elif task == "Profiles":
                st.subheader("평가 기록")
                st.write('0 : 매우적음, 4 : 매우 많음')
                user_result = view_skin_analysis(username)
                c.close()
                clean_db = pd.DataFrame(user_result,columns=['wrinkle','oil','sensative','pigmentation','hydration'])
                st.dataframe(clean_db)
                with st.expander("그래프로 확인하기"):
                    st.line_chart(clean_db)
                #user_result = view_all_users()
                #clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                #st.dataframe(clean_db)
        else:
            st.warning("아이디 또는 비밀번호가 맞지 않습니다.")

elif choice == "SignUp":
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
            st.info("Go to Login Menu to login")





   

