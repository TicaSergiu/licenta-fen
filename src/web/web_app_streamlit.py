import webbrowser

import numpy as np
import streamlit as st
from transformer.get_fen import get_fen_from_img
import cv2

st.set_page_config(page_title='ChessAI', page_icon='♟️', layout='wide')
col1, col2 = st.columns([2, 1])


def salveaza_stare():
    if 'img' not in st.session_state:
        st.session_state.img = imagine_incarcata
    if 'culoare' not in st.session_state:
        st.session_state.culoare = culoare
    if 'fen' not in st.session_state:
        st.session_state.fen = fen_rezultat
    if 'btn' not in st.session_state:
        st.session_state.btn = btn_lichess


def refoloseste_stare():
    if 'img' in st.session_state:
        global imagine_incarcata
        imagine_incarcata = st.session_state.img
        with col2:
            st.image(imagine_incarcata)
    if 'culoare' in st.session_state:
        global culoare
        culoare = st.session_state.culoare


def afiseaza_fen():
    img = imagine_incarcata.getvalue()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    fen = get_fen_from_img(img, culoare == "Alb")
    fen_rezultat.text_input('FEN', fen, disabled=True)
    if culoare == 'Negru':
        fen += "?color=black"
    btn_lichess.button('Go to lichess', on_click=lambda: webbrowser.open(f'https://lichess.org/editor/{fen}'))


if __name__ == '__main__':
    refoloseste_stare()
    with col1:
        with st.form('my-form'):
            imagine_incarcata = st.file_uploader('Incarca imagine', type=['png', 'jpg', 'jpeg'])
            culoare = st.radio('Culoare', ['Alb', 'Negru'])
            submit = st.form_submit_button('Află FEN', type='primary')
        asteptare = st.empty()
        fen_rezultat = st.empty()
        btn_lichess = st.empty()

    match [submit, imagine_incarcata]:
        case [True, None]:
            with col2:
                st.warning('Trebuie să încărcați o imagine')
        case [True, _]:
            salveaza_stare()
            with col2:
                st.image(imagine_incarcata)
            with col1:
                afiseaza_fen()
