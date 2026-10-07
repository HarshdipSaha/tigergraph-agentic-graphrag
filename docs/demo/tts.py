"""Generate narration WAV files via Windows SAPI (offline, no API keys, no network)."""
from pathlib import Path

import win32com.client

SCENES = {
    "scene1": "Investigation Certificates for Agentic GraphRAG. Built for the TigerGraph Agentic GraphRAG Hackathon.",
    "scene2": "Every answer here ships a certificate. Proof, straight from the graph's own structure, that the evidence is as complete as the question needs. Not just that the answer looks right.",
    "scene3": "On the same one hundred public questions and refreshed TigerGraph graph, template routing matched ninety nine answers. The LLM planner matched ninety eight. The planner chose a graph tool for eighty six questions.",
    "scene4": "The planner asks the model to choose a typed tool and arguments. Python validates the action, runs the graph query, and checks the returned evidence. Exhaustive aggregation and superlative results matched all thirty one public answers.",
    "scene5": "ExCeL on 30 July has three exact-date Events, across fencing and judo. The question gives no sport, so the planner returns no answer and the certificate says unverified. It does not treat a model guess as proof.",
    "scene6": "The refreshed graph contains two thousand one hundred eighty seven Event records, matching the parsed corpus. Eval zero zero one now resolves to the exact tennis Event. The offline suite contains one hundred twenty passing tests; the report records the live integration result separately.",
}

speaker = win32com.client.Dispatch("SAPI.SpVoice")
for v in speaker.GetVoices():
    if "Zira" in v.GetDescription():
        speaker.Voice = v
        break
speaker.Rate = -1  # slightly slower than default for clarity

file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
SSFMCreateForWrite = 3
SAFT44kHz16BitStereo = 39

for name, text in SCENES.items():
    out = Path(__file__).parent / "audio" / f"{name}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    file_stream.Format.Type = SAFT44kHz16BitStereo
    file_stream.Open(str(out), SSFMCreateForWrite, False)
    speaker.AudioOutputStream = file_stream
    speaker.Speak(text)
    file_stream.Close()
    print("wrote", out.name)
