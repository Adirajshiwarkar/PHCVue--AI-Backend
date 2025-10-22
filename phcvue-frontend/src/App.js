import React, { useRef, useState, useCallback, useEffect } from "react";

// --- OPTIMIZED CONFIGURATION FOR FAST BARGE-IN ---
const WS_URL = "ws://localhost:8000/phcvue";
const REQUESTED_UPLINK_SAMPLE_RATE = 16000;
const DEFAULT_TTS_SAMPLE_RATE = 24000;
const PLAYBACK_BUFFER_TARGET_SECONDS = 0.2; // Reduced from 0.4 for faster response

// --- VOICE ACTIVITY DETECTION (VAD) CONFIGURATION ---
const VOICE_ACTIVITY_THRESHOLD = 0.02;
const SILENCE_DURATION_MS = 600; // Reduced from 800ms
const ENERGY_CONSISTENCY_WINDOW = 4; // Reduced from 5
const MIN_ENERGY_RATIO = 0.5; // Reduced from 0.6
const SPECTRAL_ROLLOFF_THRESHOLD = 0.85;
const ZCR_THRESHOLD = 0.3;

export default function App() {
  const socketRef = useRef(null);
  const conversationIdRef = useRef(null);

  // Input capture refs
  const inputAudioContextRef = useRef(null);
  const inputAudioWorkletNodeRef = useRef(null);
  const streamRef = useRef(null);

  // Voice Activity Detection refs
  const voiceActivityRef = useRef(false);
  const isAIRespondingRef = useRef(false);
  const silenceTimeoutRef = useRef(null);

  // Advanced VAD state
  const energyHistoryRef = useRef([]);
  const backgroundNoiseRef = useRef(0.001);
  const speechConfidenceRef = useRef(0);

  // Playback Logic Refs
  const audioCtxRef = useRef(null);
  const playbackQueueRef = useRef([]);
  const nextStartTimeRef = useRef(0);
  const isPlayingRef = useRef(false);
  const currentAudioSourceRef = useRef(null);
  const serverFormatRef = useRef({
    codec: "pcm_s16le",
    sampleRate: DEFAULT_TTS_SAMPLE_RATE,
    channels: 1,
  });

  // Reusable audio chain nodes
  const masterGainRef = useRef(null);
  const compressorRef = useRef(null);
  const filterRef = useRef(null);

  // Debug refs
  const chunkCountRef = useRef(0);
  const totalBytesRef = useRef(0);
  const bufferedDurationRef = useRef(0);

  // UI state
  const [isRecording, setIsRecording] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("disconnected");
  const [conversationId, setConversationId] = useState(null);
  const [error, setError] = useState(null);
  const [isAIResponding, setIsAIResponding] = useState(false);
  const [voiceActivity, setVoiceActivity] = useState(false);
  const [speechConfidence, setSpeechConfidence] = useState(0);
  // --- ADDED ---
  const [language, setLanguage] = useState("en-US"); // State for language selection

  // Optimized AudioContext creation with better reuse
  const ensureAudioContext = useCallback(() => {
    if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
      try {
        console.log("Creating new AudioContext for playback...");
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)({
          sampleRate: serverFormatRef.current.sampleRate,
          latencyHint: "interactive", // Prioritize low latency
        });

        masterGainRef.current = audioCtxRef.current.createGain();
        compressorRef.current = audioCtxRef.current.createDynamicsCompressor();
        filterRef.current = audioCtxRef.current.createBiquadFilter();

        // Optimized audio processing settings
        masterGainRef.current.gain.value = 0.9;
        filterRef.current.type = "highpass";
        filterRef.current.frequency.value = 120;
        compressorRef.current.threshold.value = -24;
        compressorRef.current.knee.value = 30;
        compressorRef.current.ratio.value = 3;
        compressorRef.current.attack.value = 0.003; // Fast attack for speech
        compressorRef.current.release.value = 0.25; // Quick release

        masterGainRef.current
          .connect(filterRef.current)
          .connect(compressorRef.current)
          .connect(audioCtxRef.current.destination);

        nextStartTimeRef.current = audioCtxRef.current.currentTime + 0.05; // Reduced buffer
        console.log(`AudioContext created with sample rate: ${audioCtxRef.current.sampleRate} Hz`);
      } catch (e) {
        console.error("Failed to create AudioContext:", e);
        setError("Web Audio API is not supported by this browser.");
      }
    }

    if (audioCtxRef.current && audioCtxRef.current.state === "suspended") {
      audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  // IMMEDIATE audio stop with multiple safety layers
  const stopAndClearPlayback = useCallback((reason = "unknown") => {
    console.log(`Stopping playback and flushing queue (reason: ${reason}).`);

    try {
      // LAYER 1: Stop current source immediately
      if (currentAudioSourceRef.current) {
        currentAudioSourceRef.current.stop(0); // Immediate stop
        currentAudioSourceRef.current.disconnect();
        currentAudioSourceRef.current = null;
      }

      // LAYER 2: Mute master gain instantly
      if (masterGainRef.current) {
        masterGainRef.current.gain.setValueAtTime(0, audioCtxRef.current?.currentTime || 0);
      }

      // LAYER 3: Close AudioContext to kill all scheduled audio
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close();
      }
    } catch (e) {
      console.log("Audio cleanup completed");
    }

    // Reset all references
    audioCtxRef.current = null;
    masterGainRef.current = null;
    compressorRef.current = null;
    filterRef.current = null;

    // Clear state
    playbackQueueRef.current = [];
    bufferedDurationRef.current = 0;
    isPlayingRef.current = false;
    isAIRespondingRef.current = false;
    nextStartTimeRef.current = 0;
    setIsAIResponding(false);
  }, []);

  // Advanced VAD helper functions
  const calculateZCR = useCallback((audioData) => {
    let crossings = 0;
    for (let i = 1; i < audioData.length; i++) {
      if ((audioData[i] >= 0) !== (audioData[i - 1] >= 0)) {
        crossings++;
      }
    }
    return crossings / audioData.length;
  }, []);

  const calculateSpectralRolloff = useCallback((audioData) => {
    const highFreqEnergy = audioData.slice(audioData.length / 2).reduce((sum, val) => sum + Math.abs(val), 0);
    const totalEnergy = audioData.reduce((sum, val) => sum + Math.abs(val), 0);
    return totalEnergy > 0 ? highFreqEnergy / totalEnergy : 0;
  }, []);

  const detectVoiceActivity = useCallback((audioData) => {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      sum += audioData[i] * audioData[i];
    }
    const rms = Math.sqrt(sum / audioData.length);

    if (!voiceActivityRef.current) {
      backgroundNoiseRef.current = backgroundNoiseRef.current * 0.95 + rms * 0.05;
    }

    const zcr = calculateZCR(audioData);
    const spectralRolloff = calculateSpectralRolloff(audioData);
    const dynamicThreshold = Math.max(VOICE_ACTIVITY_THRESHOLD, backgroundNoiseRef.current * 3);

    const energyCriteria = rms > dynamicThreshold;
    const zcrCriteria = zcr < ZCR_THRESHOLD;
    const spectralCriteria = spectralRolloff < SPECTRAL_ROLLOFF_THRESHOLD;

    energyHistoryRef.current.push(rms > dynamicThreshold ? 1 : 0);
    if (energyHistoryRef.current.length > ENERGY_CONSISTENCY_WINDOW) {
      energyHistoryRef.current.shift();
    }
    const consistencyRatio = energyHistoryRef.current.reduce((sum, val) => sum + val, 0) / energyHistoryRef.current.length;

    let confidence = 0;
    if (energyCriteria) confidence += 0.4;
    if (zcrCriteria) confidence += 0.3;
    if (spectralCriteria) confidence += 0.2;
    if (consistencyRatio >= MIN_ENERGY_RATIO) confidence += 0.1;

    speechConfidenceRef.current = confidence;
    setSpeechConfidence(confidence);

    const hasVoice = energyCriteria && zcrCriteria && spectralCriteria && consistencyRatio >= MIN_ENERGY_RATIO;

    if (hasVoice) {
      if (!voiceActivityRef.current) {
        voiceActivityRef.current = true;
        setVoiceActivity(true);

        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
          silenceTimeoutRef.current = null;
        }
      }
    } else if (voiceActivityRef.current) {
      if (!silenceTimeoutRef.current) {
        silenceTimeoutRef.current = setTimeout(() => {
          voiceActivityRef.current = false;
          setVoiceActivity(false);
          energyHistoryRef.current = [];
          silenceTimeoutRef.current = null;
        }, SILENCE_DURATION_MS);
      }
    }
  }, [calculateZCR, calculateSpectralRolloff]);

  const int16ToFloat32 = useCallback((int16) => {
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      const s = int16[i];
      float32[i] = s < 0 ? s / 32768.0 : s / 32767.0;
    }
    return float32;
  }, []);

  // Optimized audio processing with smaller chunks
  const processAndPlayQueue = useCallback(() => {
    const audioCtx = ensureAudioContext();
    if (!audioCtx || playbackQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }

    // Process smaller chunks for better responsiveness
    const maxSamplesToProcess = audioCtx.sampleRate * 1.5; // Max 1.5 seconds
    const chunksToProcess = [];
    let totalSamples = 0;

    for (const chunk of playbackQueueRef.current) {
      if (totalSamples + chunk.samples.length > maxSamplesToProcess) {
        break;
      }
      chunksToProcess.push(chunk);
      totalSamples += chunk.samples.length;
    }

    if (chunksToProcess.length === 0) return;

    // Remove processed chunks
    playbackQueueRef.current = playbackQueueRef.current.slice(chunksToProcess.length);

    // Consolidate samples
    const consolidatedSamples = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of chunksToProcess) {
      consolidatedSamples.set(chunk.samples, offset);
      offset += chunk.samples.length;
    }

    const sampleRate = chunksToProcess[0].sampleRate;
    bufferedDurationRef.current -= (totalSamples / sampleRate) * 1000;

    try {
      const buffer = audioCtx.createBuffer(1, consolidatedSamples.length, sampleRate);
      buffer.copyToChannel(consolidatedSamples, 0, 0);

      const source = audioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(masterGainRef.current);
      currentAudioSourceRef.current = source;

      const now = audioCtx.currentTime;
      const startTime = Math.max(now, nextStartTimeRef.current);

      source.start(startTime);
      nextStartTimeRef.current = startTime + buffer.duration;

      if (!isAIRespondingRef.current) {
        isAIRespondingRef.current = true;
        setIsAIResponding(true);
      }

      source.onended = () => {
        if (currentAudioSourceRef.current === source) {
          currentAudioSourceRef.current = null;
        }
        isPlayingRef.current = false;

        if (playbackQueueRef.current.length > 0) {
          setTimeout(() => processAndPlayQueue(), 5); // Small delay to prevent tight loop
        } else {
          isAIRespondingRef.current = false;
          setIsAIResponding(false);
        }
      };
    } catch (err) {
      console.error("Error scheduling audio buffer:", err);
      isPlayingRef.current = false;
      isAIRespondingRef.current = false;
      setIsAIResponding(false);
    }
  }, [ensureAudioContext]);

  const handleIncomingPcm16 = useCallback((arrayBuffer) => {
    chunkCountRef.current++;
    totalBytesRef.current += arrayBuffer.byteLength;

    if (arrayBuffer.byteLength % 2 !== 0) {
      arrayBuffer = arrayBuffer.slice(0, arrayBuffer.byteLength - 1);
    }

    const int16 = new Int16Array(arrayBuffer);
    const float32 = int16ToFloat32(int16);
    const sampleRate = serverFormatRef.current.sampleRate;
    const chunkDurationMs = (float32.length / sampleRate) * 1000;

    playbackQueueRef.current.push({ samples: float32, sampleRate });
    bufferedDurationRef.current += chunkDurationMs;

    const targetBufferMs = PLAYBACK_BUFFER_TARGET_SECONDS * 1000;
    if (bufferedDurationRef.current >= targetBufferMs && !isPlayingRef.current) {
      isPlayingRef.current = true;
      processAndPlayQueue();
    }
  }, [int16ToFloat32, processAndPlayQueue]);

  // WebSocket connection with optimized message handling
  const connectWS = useCallback((convId) => {
    return new Promise((resolve, reject) => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }
      setConnectionStatus("connecting");
      const ws = new WebSocket(WS_URL);
      socketRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        setConnectionStatus("connected");
        // --- MODIFIED ---
        const startMessage = {
          type: "start",
          config: {
            language_code: language, // Send selected language
            conversation_id: convId
          }
        };
        ws.send(JSON.stringify(startMessage));
        resolve();
      };

      // OPTIMIZED message handler with clear command priority
      ws.onmessage = (event) => {
        if (typeof event.data === "string") {
          try {
            const parsed = JSON.parse(event.data);

            // HIGHEST PRIORITY: Handle clear command IMMEDIATELY
            if (parsed?.type === "clear") {
              console.log("Received 'clear' command. Stopping playback and acknowledging.");

              // Stop audio FIRST, then send ack
              stopAndClearPlayback("server_clear_command");

              if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: "clear_ack" }));
              }
              return; // Exit immediately
            }

            // Handle other message types
            if (parsed?.type === "format") {
              const newSampleRate = parsed.sampleRate || DEFAULT_TTS_SAMPLE_RATE;
              serverFormatRef.current = {
                ...serverFormatRef.current,
                ...parsed,
                sampleRate: newSampleRate
              };
            } else if (parsed?.type === "response_end") {
              isAIRespondingRef.current = false;
              setIsAIResponding(false);
            }

          } catch (e) {
            console.warn("Failed to parse server JSON", e);
          }
        } else if (event.data instanceof ArrayBuffer) {
          // Only process audio if context exists
          if (audioCtxRef.current || ensureAudioContext()) {
            handleIncomingPcm16(event.data);
          }
        }
      };

      ws.onerror = (err) => {
        setConnectionStatus("error");
        setError("WebSocket error.");
        reject(err);
      };

      ws.onclose = () => {
        setConnectionStatus("disconnected");
        setIsRecording(false);
        stopAndClearPlayback("websocket_close");
      };
    });
  // --- MODIFIED ---
  }, [language, handleIncomingPcm16, stopAndClearPlayback, ensureAudioContext]);

  const startRecording = useCallback(async () => {
    if (isRecording) return;
    setError(null);

    try {
      const newConversationId = "68d12c3c8e30a18899e137df"
      conversationIdRef.current = newConversationId;
      setConversationId(newConversationId);

      await connectWS(newConversationId);

      const audioCtx = inputAudioContextRef.current || new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: REQUESTED_UPLINK_SAMPLE_RATE
      });
      inputAudioContextRef.current = audioCtx;

      // Optimized Audio Worklet with smaller buffer sizes
      const inputWorkletCode = `
        class AudioProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.bufferSize = 1024; // Smaller for faster processing
            this._buffer = new Float32Array(0);
          }

          process(inputs) {
            const channel = inputs[0]?.[0];
            if (!channel) return true;

            // Immediate voice activity detection
            this.port.postMessage({
              type: 'voice_activity_data',
              audioData: Array.from(channel),
              timestamp: currentTime
            });

            const newBuf = new Float32Array(this._buffer.length + channel.length);
            newBuf.set(this._buffer, 0);
            newBuf.set(channel, this._buffer.length);
            this._buffer = newBuf;

            while (this._buffer.length >= this.bufferSize) {
              const chunk = this._buffer.subarray(0, this.bufferSize);
              const int16 = new Int16Array(chunk.length);

              for (let i = 0; i < chunk.length; i++) {
                let s = Math.max(-1, Math.min(1, chunk[i]));
                int16[i] = s < 0 ? s * 32768 : s * 32767;
              }

              this.port.postMessage({
                type: 'audio_data',
                buffer: int16.buffer
              }, [int16.buffer]);

              this._buffer = this._buffer.subarray(this.bufferSize);
            }

            return true;
          }
        }
        registerProcessor('audio-processor', AudioProcessor);
      `;

      if (audioCtx.audioWorklet.addModule) {
        await audioCtx.audioWorklet.addModule(
          URL.createObjectURL(new Blob([inputWorkletCode], { type: "application/javascript" }))
        );
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: REQUESTED_UPLINK_SAMPLE_RATE
        }
      });
      streamRef.current = stream;

      const source = audioCtx.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioCtx, "audio-processor");

      // Smaller frame size for faster response
      const frameMs = 10; // Reduced from 20
      const bufferSize = Math.floor((source.context.sampleRate * frameMs) / 1000);
      workletNode.port.postMessage({ type: "config", bufferSize });

      workletNode.port.onmessage = (event) => {
        if (event.data.type === 'voice_activity_data') {
          detectVoiceActivity(event.data.audioData);
        } else if (event.data.type === 'audio_data') {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(event.data.buffer);
          }
        }
      };

      source.connect(workletNode);
      inputAudioWorkletNodeRef.current = workletNode;

      setIsRecording(true);
      if (audioCtx.state === "suspended") await audioCtx.resume();

    } catch (err) {
      setError(err.message || "Could not start recording. Check mic permissions.");
      setIsRecording(false);
    }
  }, [isRecording, connectWS, detectVoiceActivity]);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    setIsRecording(false);

    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify({ type: "stop" }));
      socketRef.current.close();
    }

    streamRef.current?.getTracks().forEach((t) => t.stop());
    inputAudioWorkletNodeRef.current?.disconnect();
    if (inputAudioContextRef.current && inputAudioContextRef.current.state !== "closed") {
      inputAudioContextRef.current.close().then(() => {
        inputAudioContextRef.current = null;
      });
    }

    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    stopAndClearPlayback("stop_recording");
  }, [isRecording, stopAndClearPlayback]);

  useEffect(() => {
    return () => stopRecording();
  }, [stopRecording]);

  const statusColors = {
    disconnected: "bg-red-100 text-red-800 border-red-200",
    error: "bg-red-100 text-red-800 border-red-200",
    connecting: "bg-yellow-100 text-yellow-800 border-yellow-200",
    connected: "bg-green-100 text-green-800 border-green-200"
  };

  const statusIcons = {
    disconnected: "🔴",
    error: "❌",
    connecting: "🟡",
    connected: "🟢"
  };

  return (
    <div className="font-sans bg-gray-50 min-h-screen p-4 sm:p-6 md:p-8">
      <div className="max-w-2xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800">🎤 Voice Assistant</h1>
          <p className="text-gray-500 mt-1">Real-time voice chat with optimized barge-in response</p>
        </header>

        <main className="bg-white p-6 rounded-xl shadow-lg border border-gray-200 space-y-6">
          <div className="flex items-center space-x-3 flex-wrap gap-2">
            <span className={`px-4 py-1.5 rounded-full font-semibold text-sm border ${statusColors[connectionStatus]}`}>
              {statusIcons[connectionStatus]} Status: {connectionStatus.charAt(0).toUpperCase() + connectionStatus.slice(1)}
            </span>
            {isAIResponding && (
              <span className="px-4 py-1.5 rounded-full font-semibold text-sm bg-blue-100 text-blue-800 border-blue-200 animate-pulse">
                🤖 AI Speaking
              </span>
            )}
            {voiceActivity && (
              <span className="px-4 py-1.5 rounded-full font-semibold text-sm bg-purple-100 text-purple-800 border-purple-200">
                🎙️ Voice Detected
              </span>
            )}
            {speechConfidence > 0.5 && (
              <span className="px-4 py-1.5 rounded-full font-semibold text-sm bg-green-100 text-green-800 border-green-200">
                📊 Confidence: {(speechConfidence * 100).toFixed(0)}%
              </span>
            )}
          </div>

          <div className="text-sm text-gray-600 bg-gray-100 p-4 rounded-lg space-y-1">
            <p><strong>Conversation ID:</strong> <span className="font-mono text-xs">{conversationId || "N/A"}</span></p>
            <p><strong>🎵 Audio Format:</strong> {serverFormatRef.current.sampleRate} Hz • {serverFormatRef.current.codec}</p>
            <p><strong>📊 Stats:</strong> Chunks: {chunkCountRef.current} • Bytes: {totalBytesRef.current.toLocaleString()} • Buffer: {bufferedDurationRef.current.toFixed(0)}ms</p>
            <p><strong>🧠 Speech Confidence:</strong> {(speechConfidenceRef.current * 100).toFixed(1)}% • <strong>BG Noise:</strong> {backgroundNoiseRef.current.toFixed(4)}</p>
          </div>

          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 rounded-md">
              <p className="font-bold">Error</p>
              <p>{error}</p>
            </div>
          )}

          {/* --- ADDED --- */}
          <div className="pt-2">
            <label htmlFor="language-select" className="block text-sm font-medium text-gray-700 mb-1">
              Language
            </label>
            <select
              id="language-select"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={isRecording}
              className="w-full px-3 py-2 text-base text-gray-700 bg-white border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-200 disabled:cursor-not-allowed"
            >
              <option value="en-US">English (US)</option>
              <option value="hi-IN">Hindi (India)</option>
              <option value="kn-IN">Telugu (India)</option>
            </select>
          </div>
          {/* --- END ADDED --- */}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            <button
              onClick={startRecording}
              disabled={isRecording || connectionStatus === "connecting"}
              className="w-full px-6 py-3 text-lg font-semibold text-white bg-blue-600 rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all flex items-center justify-center"
            >
              {isRecording ? (
                <>
                  <span className="animate-pulse mr-2">🔴</span> Recording...
                </>
              ) : (
                "🎤 Start Chat"
              )}
            </button>
            <button
              onClick={stopRecording}
              disabled={!isRecording}
              className="w-full px-6 py-3 text-lg font-semibold text-white bg-red-600 rounded-lg shadow-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:bg-gray-400 disabled:cursor-not-allowed transition-all"
            >
              ⏹️ Stop Chat
            </button>
          </div>
        </main>
      </div>
    </div>
  );
}