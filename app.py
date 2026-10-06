import os
import time
import tempfile

import streamlit as st
import speech_recognition as sr

from dotenv import load_dotenv
from google import genai
from gtts import gTTS

from rag import retrieve_documents


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Student Support Chatbot",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI-Powered Student Support Chatbot")

st.write(
    "Ask questions about college information using text or voice."
)


# ============================================================
# CHECK API KEY
# ============================================================

if not API_KEY:

    st.error(
        "Google API key is missing. "
        "Please add GOOGLE_API_KEY to your .env file."
    )

    st.stop()


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_file):

    temp_audio_path = None

    try:

        # Save Streamlit recorded audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_audio:

            temp_audio.write(
                audio_file.getvalue()
            )

            temp_audio_path = temp_audio.name


        # Create speech recognizer
        recognizer = sr.Recognizer()


        # Open recorded WAV file
        with sr.AudioFile(temp_audio_path) as source:

            audio_data = recognizer.record(
                source
            )


        # Convert speech to text
        text = recognizer.recognize_google(
            audio_data,
            language="en-IN"
        )


        return text


    except sr.UnknownValueError:

        st.error(
            "❌ I could not understand your speech. "
            "Please speak clearly and try again."
        )

        return None


    except sr.RequestError as e:

        st.error(
            f"❌ Speech recognition service error: {e}"
        )

        return None


    except Exception as e:

        st.error(
            f"❌ Speech-to-text error: {e}"
        )

        return None


    finally:

        # Delete temporary audio file
        if temp_audio_path:

            try:

                os.remove(
                    temp_audio_path
                )

            except:

                pass


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    try:

        tts = gTTS(
            text=text,
            lang="en",
            slow=False
        )


        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )


        tts.save(
            audio_file.name
        )


        return audio_file.name


    except Exception as e:

        st.error(
            f"Text-to-speech error: {e}"
        )

        return None


# ============================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# VOICE INPUT
# ============================================================

st.subheader("🎤 Ask using your voice")

audio_input = st.audio_input(
    "Click here and speak your question"
)


# ============================================================
# TEXT INPUT
# ============================================================

question = st.chat_input(
    "Or type your question here..."
)


# ============================================================
# PROCESS VOICE INPUT
# ============================================================

if audio_input is not None and question is None:

    with st.spinner(
        "🎧 Converting your speech to text..."
    ):

        voice_question = speech_to_text(
            audio_input
        )


    if voice_question:

        st.success(
            f"🎤 You said: **{voice_question}**"
        )

        question = voice_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # ADD USER QUESTION TO CHAT HISTORY
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    results = retrieve_documents(
        question,
        top_k=3
    )


    # --------------------------------------------------------
    # IF NO RELEVANT INFORMATION
    # --------------------------------------------------------

    if not results:

        answer = (
            "Sorry, I couldn't find relevant information "
            "in the college documents. Please contact the "
            "college administration or faculty."
        )


    # --------------------------------------------------------
    # IF RELEVANT INFORMATION IS FOUND
    # --------------------------------------------------------

    else:

        context = "\n\n".join(

            f"Source: {result['source']}\n"
            f"{result['content']}"

            for result in results
        )


        # ----------------------------------------------------
        # GEMINI PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are an AI-powered student support chatbot.

Answer the student's question using the college
information provided below.

IMPORTANT RULES:

1. Use the provided college information as your main source.

2. Do not invent college rules or information.

3. If the information is not available, clearly say that
   it is not available in the college documents.

4. Give a simple and helpful answer.

5. Do not provide information that is not supported by
   the college documents.

COLLEGE INFORMATION:

{context}

STUDENT QUESTION:

{question}
"""


        # ----------------------------------------------------
        # GEMINI GENERATION WITH RETRY
        # ----------------------------------------------------

        answer = None


        for attempt in range(3):

            try:

                response = client.models.generate_content(

                    model="gemini-3.8-flash",

                    contents=prompt
                )


                answer = response.text

                break


            except Exception as e:

                error_message = str(e)


                if "503" in error_message:

                    if attempt < 2:

                        st.warning(
                            f"Gemini is busy. "
                            f"Retrying... ({attempt + 1}/2)"
                        )

                        time.sleep(5)


                    else:

                        answer = (
                            "Gemini is currently experiencing "
                            "high demand. Please try again later."
                        )


                else:

                    answer = (
                        f"Unable to generate an answer: {e}"
                    )

                    break


        # ----------------------------------------------------
        # FALLBACK
        # ----------------------------------------------------

        if not answer:

            answer = (
                "Gemini is temporarily unavailable. "
                "Please try again in a few seconds."
            )


    # ========================================================
    # DISPLAY AI RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # TEXT TO SPEECH
        # ----------------------------------------------------

        with st.spinner(
            "🔊 Creating voice response..."
        ):

            speech_file = text_to_speech(
                answer
            )


        if speech_file:

            st.audio(
                speech_file,
                format="audio/mp3"
            )


    # ========================================================
    # SAVE AI RESPONSE TO CHAT HISTORY
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )