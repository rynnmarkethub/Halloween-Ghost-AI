import os
import tempfile
import traceback
import google.generativeai as genai
import pyttsx3
import speech_recognition as sr
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import time
import importlib


#Gemini Setup
api_key = os.getenv("GENAI_API_KEY")
if not api_key:
   raise ValueError("Error: GENAI_API_KEY not set.")
genai.configure(api_key=api_key)


# Use a widely available model name to avoid 404s
model = genai.GenerativeModel('gemini-2.5-pro')
chat_session = model.start_chat(history=[{
   "role": "user",
   "parts": (
       "From now on, you are a ghost haunting a house on Halloween night. "
       "You have returned from the dead to greet trick-or-treaters at your doorstep. "
       "Speak in a threatening, dramatic, and haunting tone with a ghost-like whisper. "
       "Keep your responses short, 1–2 sentences, and make them chilling, "
       "mysterious, and playful enough to frighten or intrigue children. "
       "Use spooky Halloween imagery and sudden dramatic pauses to enhance the eerie effect."
   )
}])


# Optional warmup
try:
   _ = chat_session.send_message("...")
except Exception:
   pass


# speech recognizer
r = sr.Recognizer()


# tts engine
_tts_engine = pyttsx3.init()
_tts_engine.setProperty('rate', 110)
_tts_engine.setProperty('volume', 0.9)


# Try to pick a deep/male voice if available
voices = _tts_engine.getProperty('voices')
for v in voices:
   name_l = (v.name or "").lower()
   id_l = (v.id or "").lower()
   if "male" in name_l or "baritone" in name_l or "david" in id_l or "alex" in id_l:
       _tts_engine.setProperty('voice', v.id)
       break


# audio processing / playback
def change_pitch(sound: AudioSegment, semitones: float) -> AudioSegment:
   # Shift pitch by semitones using sample-rate trick
   if semitones == 0:
       return sound
   new_sample_rate = int(sound.frame_rate * (2.0 ** (semitones / 12.0)))
   pitched = sound._spawn(sound.raw_data, overrides={"frame_rate": new_sample_rate})
   return pitched.set_frame_rate(44100)


def apply_ghost_effect(segment: AudioSegment,
                      pitch_semitones: float = -3.0,
                      echo_delays=(180, 420),
                      echo_gains=(-8, -12),
                      low_pass_cutoff=4000,
                      fade_out_ms=700) -> AudioSegment:
   # Apply pitch shift, layered echoes, reversed overlay, low-pass filter and fade.
   pitched = change_pitch(segment, pitch_semitones)
   base = pitched - 2  # quieter base so echoes stand out


   combined = base
   for delay_ms, gain_db in zip(echo_delays, echo_gains):
       echo = base + gain_db
       combined = combined.overlay(echo, position=delay_ms)


   rev_copy = combined.reverse().fade_in(60).fade_out(120).reverse()
   combined = combined.overlay(rev_copy - 16)


   ghostly = combined.low_pass_filter(low_pass_cutoff)
   ghostly = ghostly.fade_out(fade_out_ms)
   return ghostly


def play_audiosegment_via_sounddevice(seg: AudioSegment):
   """Play a pydub AudioSegment via sounddevice (numpy)."""
   samples = np.array(seg.get_array_of_samples())


   # reshape for stereo
   if seg.channels == 2:
       samples = samples.reshape((-1, 2))


   # ensure int16 dtype for sounddevice when using integer samples
   if samples.dtype != np.int16:
       samples = samples.astype(np.int16)


   sd.play(samples, samplerate=seg.frame_rate)
   sd.wait()


def _generate_tts_wav_sync(text: str) -> str:
   """
   Reinitialize pyttsx3 per call and save to a temp WAV file.
   This avoids some pyttsx3 state/driver problems observed after repeated runAndWait() calls.
   """
   # create a fresh engine each time to avoid internal state issues
   importlib.reload(pyttsx3) #workaround
   engine = pyttsx3.init()
   engine.setProperty('rate', 110)
   engine.setProperty('volume', 0.9)


   # reapply voice selection heuristics on the fresh engine
   try:
       voices = engine.getProperty('voices')
       for v in voices:
           name_l = (v.name or "").lower()
           id_l = (v.id or "").lower()
           if "male" in name_l or "baritone" in name_l or "david" in id_l or "alex" in id_l:
               engine.setProperty('voice', v.id)
               break
   except Exception:
       # non-fatal; continue with default voice
       pass


   fd, path = tempfile.mkstemp(suffix=".wav")
   print(path)
   os.close(fd)
   try:
       engine.save_to_file(text, path)
       engine.runAndWait()
   finally:
       # explicitly stop & delete engine to free resources
       try:
           engine.stop()
           del(engine)
       except Exception:
           pass
   if not os.path.exists(path) or os.path.getsize(path) == 0:
       # strong indicator something went wrong writing the file
       raise RuntimeError(f"TTS did not produce a valid WAV at {path}")
   return path


def speak(text: str,
         pitch_semitones: float = -3.0,
         echo_delays=(180, 420),
         echo_gains=(-8, -12)):
   """Synthesize text, apply ghost effects, and play (synchronous and more-robust)."""
   try:
       if not text:
           print("👻 speak() called with empty text — nothing to say.")
           return


       print(f"👻 Bot: {text!r}")
       tmp_path = None
       tmp_path = _generate_tts_wav_sync(text)


       # debug: ensure file exists and is non-empty
       print("Generated TTS WAV:", tmp_path, "size:", os.path.getsize(tmp_path))


       seg = AudioSegment.from_wav(tmp_path)


       ghost_audio = apply_ghost_effect(seg,
                                       pitch_semitones=pitch_semitones,
                                       echo_delays=echo_delays,
                                       echo_gains=echo_gains)


       # debug: show some properties before playing
       print(f"Playing audio: {len(ghost_audio)} ms, channels={ghost_audio.channels}, frame_rate={ghost_audio.frame_rate}")
       play_audiosegment_via_sounddevice(ghost_audio)


   except Exception as e:
       print("Error in speak():", str(e))
       traceback.print_exc()
       # fallback attempt: try a simpler synchronous TTS speak using a fresh engine
       try:
           print("Attempting fallback pyttsx3.say()")
           fallback_engine = pyttsx3.init()
           fallback_engine.say(text or "...")
           fallback_engine.runAndWait()
           fallback_engine.stop()
       except Exception as e2:
           print("Fallback TTS also failed:", e2)
           traceback.print_exc()
   finally:
       # cleanup the temp file
       try:
           if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
               os.remove(tmp_path)
       except Exception:
           pass


# ------------------ Listening ------------------
def listen_for_speech():
   """More verbose listen_for_speech: prints entry/exit and any exceptions."""
   print("listen_for_speech(): entering")
   try:
       with sr.Microphone() as source:
           print("🎤 Microphone opened, adjusting for ambient noise...")
           r.adjust_for_ambient_noise(source, duration=0.4)
           r.pause_threshold = 0.5
           print("🎧 Listening now (phrase_time_limit=4)...")
           audio = r.listen(source, phrase_time_limit=4)
           print("🎧 Recording captured, sending to recognizer...")
   except Exception as e:
       print("listen_for_speech(): Microphone error:", repr(e))
       traceback.print_exc()
       # Return None to indicate a hardware/permission problem
       return None


   try:
       command = r.recognize_google(audio)
       print(f"🗣️ You said: {command!r}")
       return command
   except sr.UnknownValueError:
       print("listen_for_speech(): UnknownValueError — speech not understood.")
       return ""
   except Exception as e:
       print("listen_for_speech(): recognition error:", repr(e))
       traceback.print_exc()
       return ""


def get_response_text(response):
   """
   Try multiple common attributes that may contain the assistant text.
   This handles different shapes of responses from the API/library.
   """
   # Inspect the object for debugging (first few runs)
   try:
       # this will print short representation of the object and its attrs
       print("DEBUG: raw response repr:", repr(response)[:1000])
   except Exception:
       pass


   # Common places text could live:
   candidates = []
   # 1) Some libs use .text
   if hasattr(response, "text") and response.text:
       return response.text
   # 2) Some libs expose .candidates (list) or .outputs
   if hasattr(response, "candidates"):
       try:
           # join all candidate texts (if present)
           for c in response.candidates:
               if hasattr(c, "content"):
                   candidates.append(getattr(c, "content"))
               elif isinstance(c, str):
                   candidates.append(c)
       except Exception:
           pass
       if candidates:
           return " ".join([c for c in candidates if c])
   if hasattr(response, "outputs"):
       try:
           for o in response.outputs:
               txt = getattr(o, "text", None) or o if isinstance(o, str) else None
               if txt:
                   candidates.append(txt)
       except Exception:
           pass
       if candidates:
           return " ".join(candidates)
   # 3) fallback to str(response)
   try:
       s = str(response)
       if s and s.strip() and s != "<empty response>":
           return s
   except Exception:
       pass
   return ""


def main():
   print("main(): starting up")
   # Debug: confirm chat_session exists
   try:
       print("main(): chat_session object repr:", repr(chat_session)[:1000])
   except Exception:
       pass


   # initial greeting
   try:
       print("main(): calling speak() for initial greeting")
       speak("What is it you want, mortal?")
       print("main(): initial speak() returned")
   except Exception as e:
       print("main(): speak() for initial greeting raised:", repr(e))
       traceback.print_exc()


   loop_count = 0
   while True:
       loop_count += 1
       print(f"main(): loop iteration {loop_count} — calling listen_for_speech()")
       user_input = listen_for_speech()
       print("main(): listen_for_speech() returned:", repr(user_input))


       # Distinguish between microphone errors (None), silence (""), and actual text
       if user_input is None:
           print("main(): Microphone likely failed or not available. Sleeping 3s then retrying.")
           time.sleep(3)
           # Try again a few times; if still None after several attempts, exit for debugging
           if loop_count >= 6:
               print("main(): microphone problems persist — exiting main() for debugging.")
               break
           continue


       if user_input == "":
           print("main(): no speech detected this round. continuing.")
           continue


       # handle stop words
       if any(word in user_input.lower() for word in ["stop", "goodbye", "bye", "thank you"]):
           print("main(): received a stop word from user_input. saying goodbye and breaking.")
           speak("Until next time, mortal...")
           break


       # send to LLM and speak the response
       try:
           print("main(): sending to chat_session.send_message() ...")
           response = chat_session.send_message(user_input)
           print("main(): raw response repr:", repr(response)[:1200])
           out_text = get_response_text(response)
           print("main(): extracted text to speak:", repr(out_text))
           if not out_text:
               print("main(): extracted text empty — will speak a fallback.")
               speak("...")
           else:
               speak(out_text)
       except Exception as e:
           print("main(): LLM/send/speak error:", repr(e))
           traceback.print_exc()
           speak("What are you rambling on about, mortal?")


   print("main(): finished (exiting normally)")


if __name__ == "__main__":
   try:
       main()
   except Exception as e:
       print("Unhandled exception in __main__:", repr(e))
       traceback.print_exc()
       raise