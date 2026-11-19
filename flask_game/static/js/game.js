(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const csrf = document.body.dataset.csrf || document.querySelector('meta[name="csrf-token"]')?.content || "";
  const isAdmin = document.body.dataset.admin === "true";
  const capture = new window.BioGuardCapture($("camera-feed"), $("capture-canvas"));
  let muted = false;
  let cameraTimer = null;
  let cameraStartedAt = 0;
  let speakerFile = null;
  let commandFile = null;
  // Mini Speech Commands contains isolated words, but a one-second browser
  // window can clip the initial /s/ or final /p/ in "stop" while the user
  // starts speaking after the button click.  The server still applies the
  // unchanged trim + one-second feature pipeline; this is command-only
  // capture slack and does not affect the speaker-check recording.
  const COMMAND_RECORDING_MS = 1500;
  let authState = "IDLE";
  let serverDiagnostics = null;
  let lastApiResponse = "none";
  let lastFrontendError = "none";

  function tone(frequency, duration = .12, type = "sine") {
    if (muted || !window.AudioContext) return;
    try {
      const context = new AudioContext();
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = type;
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(.0001, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(.08, context.currentTime + .015);
      gain.gain.exponentialRampToValueAtTime(.0001, context.currentTime + duration);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + duration + .03);
      window.setTimeout(() => context.close(), (duration + .1) * 1000);
    } catch (_) {
      // Sound is decorative and never affects verification.
    }
  }

  function setAuthState(state) {
    authState = state;
    $("auth-state").textContent = state;
    $("auth-state").dataset.state = state;
    renderDiagnostics();
  }

  function setLoading(active) {
    const panel = document.querySelector(".verification-panel");
    if (panel) {
      panel.classList.toggle("loading", active);
      panel.setAttribute("aria-busy", active ? "true" : "false");
    }
  }

  function pause(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  function setTimer(active) {
    if (cameraTimer) window.clearInterval(cameraTimer);
    cameraTimer = null;
    if (!active) {
      $("camera-timer").textContent = "00:00";
      return;
    }
    cameraStartedAt = Date.now();
    cameraTimer = window.setInterval(() => {
      const seconds = Math.floor((Date.now() - cameraStartedAt) / 1000);
      $("camera-timer").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
    }, 500);
  }

  function setStep(step, state = "active") {
    document.querySelectorAll(".progress-step").forEach((element) => {
      if (element.dataset.step === step) element.classList.add(state);
      else if (state === "active" && element.classList.contains("active")) element.classList.remove("active");
    });
  }

  function setPhase(phase, title, detail) {
    $("verification-phase").textContent = phase;
    $("verification-title").textContent = title;
    $("verification-detail").textContent = detail;
  }

  function recordFrontendError(error) {
    lastFrontendError = error?.message || String(error || "Unknown frontend error");
    renderDiagnostics();
  }

  function renderDiagnostics() {
    if (!isAdmin || !$("diagnostics-panel")) return;
    const panel = $("diagnostics-panel");
    const body = $("diagnostics-body");
    const data = {
      enrolled_users: serverDiagnostics?.enrolled_users ?? "—",
      face_candidate: serverDiagnostics?.face_candidate ?? "—",
      face_score: serverDiagnostics?.face_score ?? "—",
      face_threshold: serverDiagnostics?.face_threshold ?? "—",
      face_method: serverDiagnostics?.face_method ?? "—",
      face_threshold_source: serverDiagnostics?.face_threshold_source ?? "—",
      voice_score: serverDiagnostics?.voice_score ?? "—",
      voice_threshold: serverDiagnostics?.voice_threshold ?? "—",
      voice_method: serverDiagnostics?.voice_method ?? "—",
      voice_threshold_source: serverDiagnostics?.voice_threshold_source ?? "—",
      user_role: serverDiagnostics?.user_role ?? "—",
      door_permission: serverDiagnostics?.door_permission ?? "—",
      current_auth_state: authState,
      last_api_response: lastApiResponse,
      last_frontend_error: lastFrontendError,
    };
    panel.hidden = false;
    body.replaceChildren();
    Object.entries(data).forEach(([label, value]) => {
      const item = document.createElement("div");
      item.className = "diagnostic-item";
      const key = document.createElement("small");
      key.textContent = label.replaceAll("_", " ");
      const valueNode = document.createElement("strong");
      if (typeof value === "number") valueNode.textContent = value.toFixed(4);
      else if (value === true) valueNode.textContent = "YES";
      else if (value === false) valueNode.textContent = "NO";
      else valueNode.textContent = String(value);
      item.append(key, valueNode);
      body.appendChild(item);
    });
  }

  function acceptApiResponse(label, response, result) {
    lastApiResponse = `${label} → ${response.status} / ${result.accepted ? "accepted" : (result.error_code || "denied")}`;
    if (result.diagnostics) serverDiagnostics = result.diagnostics;
    renderDiagnostics();
  }

  async function readJson(response, label) {
    try {
      const result = await response.json();
      if (!response.ok && !result.error_code) result.error_code = "SERVER_ERROR";
      acceptApiResponse(label, response, result);
      return result;
    } catch (error) {
      recordFrontendError(error);
      lastApiResponse = `${label} → ${response.status} / invalid JSON`;
      renderDiagnostics();
      return {
        accepted: false,
        error_code: response.status >= 500 ? "SERVER_ERROR" : "AUDIO_UPLOAD_FAILED",
        failure_stage: "technical",
        detail: response.status >= 500 ? "The local server returned an invalid error response." : "The audio upload did not return a valid response.",
        message: "Access Denied - request failed.",
      };
    }
  }

  async function loadAdminDiagnostics() {
    if (!isAdmin) return;
    try {
      const response = await fetch("/api/admin/diagnostics", { headers: { Accept: "application/json" } });
      serverDiagnostics = await response.json();
      lastApiResponse = `GET /api/admin/diagnostics → ${response.status}`;
      renderDiagnostics();
    } catch (error) {
      recordFrontendError(error);
    }
  }

  function showResult(pass, message) {
    const banner = $("result-banner");
    banner.hidden = false;
    banner.className = `result-banner ${pass ? "pass" : "fail"}`;
    banner.textContent = message;
    $("retry-button").hidden = false;
    $("verify-button").disabled = true;
    $("start-voice").hidden = true;
    $("door-card").classList.toggle("open", pass);
    $("door-card").classList.toggle("denied", !pass);
    $("door-state").textContent = pass ? "OPEN" : "LOCKED";
    $("door-state").className = pass ? "state-open" : "state-denied";
    $("door-message").textContent = pass ? "ACCESS GRANTED" : "ACCESS DENIED";
    $("door-message").style.color = pass ? "var(--green)" : "var(--red)";
  }

  const VOICE_FAILURE_COPY = {
    AUDIO_CAPTURE_FAILED: { title: "Microphone capture failed.", detail: "The browser did not produce a recording. Check microphone permission and try again." },
    AUDIO_DECODE_FAILED: { title: "Audio could not be decoded.", detail: "Use a supported WAV/WebM recording or record again." },
    AUDIO_UPLOAD_FAILED: { title: "Audio upload failed.", detail: "The recording did not reach the local server correctly. Try again." },
    AUDIO_TOO_SHORT: { title: "Recording too short.", detail: "Record the complete voice phrase for at least the required duration." },
    AUDIO_SILENT: { title: "No usable speech detected.", detail: "Move closer to the microphone, speak clearly, and record again." },
    VOICE_LOW_CONFIDENCE: { title: "Voice recorded, but confidence was too low.", detail: "The speaker score was below the calibrated threshold. This is a biometric rejection, not a recording failure." },
    VOICE_MISMATCH: { title: "Voice does not match the face.", detail: "The recording was processed, but it belongs to a different speaker than the accepted face candidate." },
    VOICE_AMBIGUOUS: { title: "Speaker was ambiguous.", detail: "The recording was processed, but the local candidates were too close to distinguish safely." },
    PERMISSION_DENIED: { title: "Door permission denied.", detail: "Face and voice matched, but this user is not authorized to open the door." },
    SERVER_ERROR: { title: "Server processing error.", detail: "The local server failed while processing the recording. Retry the attempt." },
  };

  function voiceFailureCopy(result) {
    const code = result?.error_code || "SERVER_ERROR";
    const copy = VOICE_FAILURE_COPY[code] || { title: "Voice verification failed.", detail: result?.detail || "The recording was rejected safely." };
    return { code, ...copy, detail: result?.detail || copy.detail };
  }

  function stopScan() {
    capture.stopCamera();
    $("camera-window").classList.remove("active");
    $("camera-placeholder").classList.remove("hidden");
    $("camera-state").textContent = "STANDBY";
    $("camera-state").className = "";
    $("start-scan").disabled = false;
    $("stop-scan").disabled = true;
    $("verify-button").disabled = true;
    setTimer(false);
  }

  async function startScan() {
    setAuthState("IDLE");
    try {
      const placeholder = $("camera-placeholder");
      placeholder.querySelector("strong").textContent = "CAMERA OFF";
      placeholder.querySelector("small").textContent = "Press START FACE SCAN to begin";
      setPhase("CAMERA_READY", "Allow camera access, then capture your face.", "Press Capture face when one face is centered in the frame.");
      await capture.startCamera();
      $("camera-window").classList.add("active");
      $("camera-placeholder").classList.add("hidden");
      $("camera-state").textContent = "LIVE";
      $("camera-state").className = "state-open";
      $("start-scan").disabled = true;
      $("stop-scan").disabled = false;
      $("verify-button").disabled = false;
      $("verify-button").hidden = false;
      $("start-voice").hidden = true;
      setAuthState("CAMERA_READY");
      setTimer(true);
      setStep("face", "active");
      tone(440, .1);
    } catch (error) {
      recordFrontendError(error);
      setAuthState("TECHNICAL_ERROR");
      setPhase("TECHNICAL ERROR", "The camera did not start.", error.message || "Allow camera access on localhost and try again.");
      showResult(false, "Access Denied - camera unavailable.");
    }
  }

  async function scanFace() {
    if (!capture.stream) return;
    $("verify-button").disabled = true;
    $("retry-button").hidden = true;
    setLoading(true);
    setAuthState("FACE_SCANNING");
    setPhase("FACE_SCANNING", "Finding your face candidate.", "The server will reject an unknown face before requesting microphone access.");
    try {
      const faceFile = await capture.snapshot();
      const form = new FormData();
      form.append("face_image", faceFile);
      const response = await fetch("/api/face-candidate", {
        method: "POST",
        headers: { "X-CSRFToken": csrf },
        body: form,
      });
      const result = await readJson(response, "POST /api/face-candidate");
      if (!result.accepted) {
        setAuthState("FACE_REJECTED");
        setStep("face", "fail");
        setPhase("FACE_REJECTED", "The face candidate was not accepted.", result.detail || "Try better lighting and keep exactly one face in frame.");
        stopScan();
        setLoading(false);
        setAuthState("ACCESS_DENIED");
        showResult(false, "ACCESS DENIED — FACE REJECTED");
        tone(170, .18, "sawtooth");
        return;
      }
      setAuthState("FACE_ACCEPTED");
      setStep("face", "pass");
      setStep("voice", "active");
      stopScan();
      $("camera-state").textContent = "FACE VERIFIED";
      $("camera-state").className = "state-open";
      $("camera-placeholder").querySelector("strong").textContent = "FACE VERIFIED";
      $("camera-placeholder").querySelector("small").textContent = "Camera off while voice is recorded";
      $("start-scan").disabled = true;
      $("verify-button").hidden = true;
      $("start-voice").hidden = false;
      $("start-voice").disabled = false;
      setPhase("FACE_ACCEPTED", "FACE VERIFIED — PLEASE RECORD YOUR VOICE", "Your face candidate passed. Press Start voice verification to continue.");
      setLoading(false);
      tone(660, .11);
    } catch (error) {
      recordFrontendError(error);
      setAuthState("TECHNICAL_ERROR");
      setStep("face", "fail");
      stopScan();
      setLoading(false);
      setPhase("TECHNICAL ERROR", "Face scanning could not complete.", error.message || "The local server failed closed. Retry the attempt.");
      showResult(false, "ACCESS DENIED — TECHNICAL ERROR");
      tone(170, .18, "sawtooth");
    }
  }

  async function startVoiceVerification() {
    $("start-voice").disabled = true;
    $("retry-button").hidden = true;
    $("voice-meter").classList.add("active");
    setLoading(true);
    setAuthState("VOICE_RECORDING");
    setPhase("VOICE_RECORDING", "Please record your voice now.", "Say \"BioGuard access\" clearly for three seconds.");
    try {
      const voiceFile = await capture.recordAudio(3000);
      $("voice-meter").classList.remove("active");
      setAuthState("VOICE_VERIFYING");
      setPhase("VOICE_VERIFYING", "Checking the same speaker.", "The server is applying the trained voice threshold and permission gate.");
      const form = new FormData();
      form.append("voice_recording", voiceFile);
      const response = await fetch("/api/verify", {
        method: "POST",
        headers: { "X-CSRFToken": csrf },
        body: form,
      });
      const result = await readJson(response, "POST /api/verify");
      setLoading(false);
      if (result.accepted) {
        setAuthState("VOICE_ACCEPTED");
        setStep("voice", "pass");
        setPhase("VOICE_ACCEPTED", "VOICE VERIFIED — CHECKING PERMISSIONS", "The same speaker matched the face candidate. Checking door permission now.");
        await pause(260);
        setAuthState("PERMISSION_CHECK");
        setStep("permission", "active");
        await pause(260);
        setAuthState("ACCESS_GRANTED");
        setStep("permission", "pass");
        setPhase("ACCESS_GRANTED", result.message || "Access granted.", `Welcome. Role: ${result.role || "authorized user"}. Door permission passed.`);
        showResult(true, `ACCESS GRANTED — WELCOME, ${result.identity || "AUTHORIZED USER"}`);
        tone(880, .12);
        window.setTimeout(() => tone(1180, .2), 110);
        $("command-section").hidden = false;
        $("camera-state").textContent = "OFF FOR COMMAND MODE";
        $("start-scan").disabled = true;
      } else {
        const failure = voiceFailureCopy(result);
        const stage = result.failure_stage || "voice";
        const technicalAudioFailure = failure.code.startsWith("AUDIO_") || failure.code === "SERVER_ERROR";
        if (stage === "permission" || failure.code === "PERMISSION_DENIED") {
          setAuthState("VOICE_ACCEPTED");
          setStep("voice", "pass");
          setAuthState("PERMISSION_CHECK");
          setStep("permission", "fail");
          setAuthState("ACCESS_DENIED");
        } else {
          setAuthState(technicalAudioFailure ? "TECHNICAL_ERROR" : "VOICE_REJECTED");
          setStep("voice", "fail");
          setStep("permission", "fail");
        }
        setPhase(failure.code, failure.title, failure.detail);
        setAuthState("ACCESS_DENIED");
        showResult(false, `ACCESS DENIED — ${failure.code}`);
        tone(170, .18, "sawtooth");
      }
    } catch (error) {
      recordFrontendError(error);
      $("voice-meter").classList.remove("active");
      setLoading(false);
      const code = error?.code || "AUDIO_UPLOAD_FAILED";
      const failure = voiceFailureCopy({ error_code: code, detail: error?.message });
      setAuthState("TECHNICAL_ERROR");
      setStep("voice", "fail");
      setStep("permission", "fail");
      setPhase(failure.code, failure.title, failure.detail);
      showResult(false, `ACCESS DENIED — ${failure.code}`);
      tone(170, .18, "sawtooth");
    }
  }

  function retry() {
    $("result-banner").hidden = true;
    $("door-card").classList.remove("open", "denied");
    $("door-state").textContent = "LOCKED";
    $("door-state").className = "state-locked";
    $("door-message").textContent = "AWAITING IDENTITY";
    $("door-message").style.color = "";
    $("retry-button").hidden = true;
    $("start-voice").hidden = true;
    $("command-section").hidden = false; // Voice-only mode is independent from door access.
    $("diagnostics-panel").hidden = !isAdmin;
    ["face", "voice", "permission"].forEach((name) => {
      const step = document.querySelector(`[data-step="${name}"]`);
      step.classList.remove("active", "pass", "fail");
    });
    setPhase("IDLE", "Start the face scan when you are ready.", "The live face is checked first; only an accepted candidate is asked for voice.");
    setAuthState("IDLE");
    startScan();
  }

  async function recordSpeaker() {
    // Explicitly switch off camera even if the user visited face scanning.
    if (capture.stream) stopScan();
    const button = $("record-speaker");
    button.disabled = true;
    $("command-state").textContent = "LISTENING";
    $("command-message").textContent = "Say the reference phrase clearly for the speaker check...";
    try {
      speakerFile = await capture.recordAudio(3000);
      $("command-message").textContent = "Speaker check captured. Now record one isolated action word.";
      $("record-command").disabled = false;
      tone(660, .1);
    } catch (error) {
      recordFrontendError(error);
      $("command-message").textContent = error.message || "Microphone unavailable.";
    } finally {
      button.disabled = false;
      $("command-state").textContent = "READY";
    }
  }

  async function recordCommand() {
    const button = $("record-command");
    button.disabled = true;
    $("command-message").textContent = "Listening for GO or STOP (1.5 seconds)...";
    try {
      commandFile = await capture.recordAudio(COMMAND_RECORDING_MS);
      $("command-message").textContent = "Command captured. Submit for speaker + permission verification.";
      $("send-command").disabled = false;
      tone(760, .1);
    } catch (error) {
      recordFrontendError(error);
      $("command-message").textContent = error.message || "Microphone unavailable.";
    } finally {
      button.disabled = false;
    }
  }

  async function sendCommand() {
    if (!speakerFile || !commandFile) return;
    const button = $("send-command");
    button.disabled = true;
    $("command-state").textContent = "VERIFYING";
    $("command-message").textContent = "Checking speaker, confidence, command word, and permission...";
    const form = new FormData();
    form.append("speaker_audio", speakerFile);
    form.append("command_audio", commandFile);
    try {
      const response = await fetch("/api/command", { method: "POST", headers: { "X-CSRFToken": csrf }, body: form });
      const result = await readJson(response, "POST /api/command");
      if (result.accepted) {
        const door = $("door-card");
        door.classList.toggle("open", result.action === "open");
        $("door-state").textContent = result.action === "open" ? "OPEN" : "LOCKED";
        $("door-state").className = result.action === "open" ? "state-open" : "state-locked";
        $("door-message").textContent = result.action === "open" ? "ACCESS GRANTED" : "DOOR SECURED";
        $("command-state").textContent = result.action === "open" ? "OPEN" : "CLOSED";
        $("command-message").textContent = `${String(result.command || "command").toUpperCase()} authorized - ${result.action} action permitted.`;
        tone(940, .16);
      } else {
        $("command-state").textContent = "REJECTED";
        $("command-message").textContent = result.detail || result.message || result.reason || "Command rejected.";
        tone(170, .18, "sawtooth");
      }
    } catch (error) {
      recordFrontendError(error);
      $("command-state").textContent = "REJECTED";
      $("command-message").textContent = "Command failed closed.";
    } finally {
      // Require a fresh speaker sample and fresh command for the next action.
      speakerFile = null;
      commandFile = null;
      $("record-command").disabled = true;
      button.disabled = true;
    }
  }

  $("start-scan").addEventListener("click", startScan);
  $("stop-scan").addEventListener("click", stopScan);
  $("verify-button").addEventListener("click", scanFace);
  $("start-voice").addEventListener("click", startVoiceVerification);
  $("retry-button").addEventListener("click", retry);
  $("record-speaker").addEventListener("click", recordSpeaker);
  $("record-command").addEventListener("click", recordCommand);
  $("send-command").addEventListener("click", sendCommand);
  $("mute-toggle").addEventListener("click", () => {
    muted = !muted;
    $("mute-toggle").textContent = muted ? "Sound off" : "Sound on";
  });
  window.addEventListener("beforeunload", () => {
    capture.stopCamera();
    if (cameraTimer) window.clearInterval(cameraTimer);
  });
  setAuthState("IDLE");
  loadAdminDiagnostics();
})();
