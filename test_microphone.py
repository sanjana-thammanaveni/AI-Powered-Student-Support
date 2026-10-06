import speech_recognition as sr

recognizer = sr.Recognizer()

# Your microphone
MICROPHONE_INDEX = 17

print("🎤 Using microphone:")
print(sr.Microphone.list_microphone_names()[MICROPHONE_INDEX])

with sr.Microphone(device_index=MICROPHONE_INDEX) as source:

    print("\n🔧 Adjusting for background noise...")
    recognizer.adjust_for_ambient_noise(source, duration=2)

    print("✅ Ready!")
    print("🎤 Speak now: What is attendance?")

    try:
        audio = recognizer.listen(
            source,
            timeout=15,
            phrase_time_limit=10
        )

        print("🔄 Converting speech to text...")

        text = recognizer.recognize_google(
            audio,
            language="en-IN"
        )

        print("✅ You said:", text)

    except sr.WaitTimeoutError:
        print("❌ No speech detected.")

    except sr.UnknownValueError:
        print("❌ Speech was detected, but Google could not understand it.")

    except sr.RequestError as e:
        print("❌ Google Speech Recognition error:", e)

    except Exception as e:
        print("❌ Error:", e)