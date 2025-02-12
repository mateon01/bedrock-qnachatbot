''' chat.py consists of all chat related functions of LOKI'''

import streamlit as st ###### Import Streamlit library
from streamlit_chat import message ###### Import message function from streamlit_chat library to render chat
import base64
from io import BytesIO
from PIL import Image


def new_img(base64_img):
    try:
        # Ensure the input is a valid base64 string
        if isinstance(base64_img, str) and base64_img.strip():
            # Attempt to decode the base64 string
            decoded_data = base64.b64decode(base64_img)
            aux_im = Image.open(BytesIO(decoded_data))
            return aux_im
        else:
            print("Invalid base64 string provided.")
            return None
    except (ValueError, UnicodeEncodeError) as e:
        print(f"Error decoding base64 string: {e}")
        return None

### Initialisation function for the chat. Takes input for the first message.
### Also initialises the history and pastinp and pastresp variables
### history is used to store the chat history
### pastinp is used to store the past user inputs
### pastresp is used to store the past bot responses
### pastinp and pastresp are used to render the chat
### pastinp and pastresp are lists of strings
### history is a list of lists of strings because the chat history is stored as a list of lists of strings
def initialize_chat(bot_m=None):
    if 'history' not in st.session_state:
        st.session_state['history']=[]

    if 'pastinp' not in st.session_state:
        st.session_state['pastinp']=[]

    if 'pastresp' not in st.session_state:
        st.session_state['pastresp']=[bot_m]

    if 'pastresp_table' not in st.session_state:
        st.session_state['pastresp_table'] = [bot_m]
        
    if 'pastresp_image' not in st.session_state:
        st.session_state['pastresp_image'] = [bot_m]

### Function for rendering chat
### Each list of strings is a conversation between the user and the bot
### The last element of the list is the latest conversation
### The first element of the list is the oldest conversation
### The function renders the chat in reverse order
### The latest conversation is rendered at the top
### The oldest conversation is rendered at the bottom
### The function uses the message function from the streamlit_chat library to render each message


def render_chat():
    for i in range(0, len(st.session_state['pastresp']) - 1):
        if st.session_state['pastresp'][len(st.session_state['pastresp']) - 1 - i]:
            message(st.session_state['pastresp'][len(
                st.session_state['pastresp']) - 1 - i])

            if i < len(st.session_state.get('pastresp_table', [])):
                table_content = st.session_state['pastresp_table'][len(st.session_state['pastresp_table']) - 1 - i]
                if table_content:
                    st.markdown(table_content)

            if i < len(st.session_state.get('pastresp_image', [])):
                image_content = st.session_state['pastresp_image'][len(st.session_state['pastresp_image']) - 1 - i]
                image = new_img(image_content)
                if image:
                    st.image(image)

        if st.session_state['pastinp'][len(st.session_state['pastinp']) - 1 - i]:
            message(st.session_state['pastinp'][len(
                st.session_state['pastinp']) - 1 - i], is_user=True)

    if st.session_state['pastresp'][0]:
        st.error(st.session_state['pastresp'][0])
        message(st.session_state['pastresp'][0])

### Function for adding the latest query and LOKI response
### to the session state variables pastinp and pastresp


def chatbot(query, info, response):
    message(query, is_user=True)
    message(response)

    if info and isinstance(info, list) and len(info) > 0:  # info가 비어 있지 않은 경우 확인
        source = info[0].get('_source', {})

        if source.get('table_md', ''):
            st.markdown(source['table_md'])

        if source.get('image_base64', ''):
            image = new_img(source['image_base64'])
            if image:
                st.image(image)
    else:
        message("No information available.")

    render_chat()
