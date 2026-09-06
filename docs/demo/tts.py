"""Generate narration WAV files via Windows SAPI (offline, no API keys, no network)."""
import win32com.client

SCENES = {
    "scene1": "Investigation Certificates for Agentic GraphRAG. Built for the TigerGraph Agentic GraphRAG Hackathon.",
    "scene2": "Every answer here ships a certificate. Proof, straight from the graph's own structure, that the evidence is as complete as the question needs. Not just that the answer looks right.",
    "scene3": "On one hundred real questions against a live TigerGraph database: plain RAG scored twenty five percent. GraphRAG, forty three. Our agentic pipeline: ninety nine percent, at forty two tokens per answer.",
    "scene4": "Here's why. Some questions need every matching event, up to forty three documents. No top-k search can retrieve that many. Both baselines score zero. Our pipeline runs a deterministic count on the graph instead, and gets every one right, for free.",
    "scene5": "And when the agent does need the language model, say, to break a tie, the certificate says so honestly, instead of hiding it.",
    "scene6": "Sixty five tests. Reconciled against the live graph. Open source on GitHub. Investigation Certificates for Agentic GraphRAG.",
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
    out = f"H:/augsepthacks/tigergraph-hack/docs/demo/audio/{name}.wav"
    file_stream.Format.Type = SAFT44kHz16BitStereo
    file_stream.Open(out, SSFMCreateForWrite, False)
    speaker.AudioOutputStream = file_stream
    speaker.Speak(text)
    file_stream.Close()
    print("wrote", out)
