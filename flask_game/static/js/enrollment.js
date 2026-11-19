(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const SLOT_COUNT = 4;
  const FACE_LABELS = ["Front face", "Slight right", "Slight left", "Different angle"];
  const VOICE_LABELS = ["Voice one", "Voice two", "Voice three", "Voice four"];
  const FACE_ACCEPT = "image/jpeg,image/png";
  const VOICE_ACCEPT = "audio/*,.wav,.webm,.ogg,.mp3,.m4a";

  const form = $("enrollment-form");
  if (!form) return;

  const faceInput = $("face-images");
  const voiceInput = $("voice-recordings");
  const cameraCapture = new window.BioGuardCapture($("enroll-camera"), $("enroll-canvas"));
  const voiceCapture = new window.BioGuardCapture(null, null);
  const draft = {
    info: {},
    faces: Array.from({ length: SLOT_COUNT }, () => emptySlot()),
    voices: Array.from({ length: SLOT_COUNT }, () => emptySlot()),
  };
  let activeFaceSlot = 0;
  let activeVoiceSlot = 0;
  let voiceTimer = null;
  let submitting = false;

  function emptySlot() {
    return { file: null, status: "empty", message: "", source: null, token: 0, previewUrl: null };
  }

  function slotsFor(kind) {
    return kind === "faces" ? draft.faces : draft.voices;
  }

  function labelsFor(kind) {
    return kind === "faces" ? FACE_LABELS : VOICE_LABELS;
  }

  function fileExtension(file) {
    return String(file?.name || "").toLowerCase().split(".").pop();
  }

  function isImageFile(file) {
    return Boolean(file && (String(file.type || "").startsWith("image/") || ["jpg", "jpeg", "png"].includes(fileExtension(file))));
  }

  function isAudioFile(file) {
    return Boolean(file && (String(file.type || "").startsWith("audio/") || ["wav", "webm", "ogg", "mp3", "m4a", "aac"].includes(fileExtension(file))));
  }

  function revokePreview(slot) {
    if (slot.previewUrl) URL.revokeObjectURL(slot.previewUrl);
    slot.previewUrl = null;
  }

  function setFormError(message) {
    const node = $("enrollment-form-error");
    if (!node) return;
    node.textContent = message || "";
    node.hidden = !message;
  }

  function setSaveStatus(message, success = false) {
    const node = $("enrollment-save-status");
    if (!node) return;
    node.textContent = message || "";
    node.classList.toggle("status-success", success);
    node.classList.toggle("status-error", Boolean(message) && !success);
  }

  function setFieldError(field, message) {
    const node = document.querySelector(`[data-field-error="${field}"]`);
    if (!node) return;
    node.textContent = message || "";
    node.hidden = !message;
    const control = form.querySelector(`[name="${field}"]`);
    if (control) control.classList.toggle("input-error", Boolean(message));
  }

  function clearFieldErrors() {
    document.querySelectorAll("[data-field-error]").forEach((node) => {
      node.textContent = "";
      node.hidden = true;
    });
    form.querySelectorAll(".input-error").forEach((node) => node.classList.remove("input-error"));
  }

  function readInfo() {
    const permissions = Array.from(form.querySelectorAll("[data-draft-permission]:checked")).map((node) => node.value);
    draft.info = {
      display_name: String(form.elements.display_name?.value || "").trim().replace(/\s+/g, " "),
      identifier: String(form.elements.identifier?.value || "").trim(),
      role: String(form.elements.role?.value || "operator"),
      active: Boolean(form.elements.active?.checked),
      permissions,
      consent: Boolean($("enrollment-consent")?.checked),
    };
    return draft.info;
  }

  function validateInfo() {
    const info = readInfo();
    const errors = {};
    if (!info.display_name || info.display_name.length > 80) errors.display_name = "Enter a name of 1-80 characters.";
    if (!/^[A-Za-z0-9_-]{2,40}$/.test(info.identifier)) errors.identifier = "Use 2-40 Latin letters, digits, underscore or hyphen characters.";
    if (!["operator", "supervisor", "security", "observer"].includes(info.role)) errors.role = "Choose a supported role.";
    if (!info.consent) errors.consent = "Consent is required before local enrollment can be saved.";
    clearFieldErrors();
    Object.entries(errors).forEach(([field, message]) => setFieldError(field, message));
    return errors;
  }

  async function validateFaceFile(file) {
    if (!isImageFile(file)) return { ok: false, message: "Choose a JPG or PNG image." };
    if (file.size <= 0) return { ok: false, message: "The image file is empty." };
    if (file.size > 12 * 1024 * 1024) return { ok: false, message: "The image is larger than 12 MB." };
    const url = URL.createObjectURL(file);
    try {
      const image = await new Promise((resolve, reject) => {
        const element = new Image();
        element.onload = () => resolve(element);
        element.onerror = () => reject(new Error("The image could not be decoded in the browser."));
        element.src = url;
      });
      if (image.width < 96 || image.height < 96) return { ok: false, message: "Use an image with at least 96 pixels on each side." };
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 64;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, 64, 64);
      const pixels = context.getImageData(0, 0, 64, 64).data;
      let sum = 0;
      let squareSum = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        const gray = (pixels[index] * 0.299) + (pixels[index + 1] * 0.587) + (pixels[index + 2] * 0.114);
        sum += gray;
        squareSum += gray * gray;
      }
      const count = pixels.length / 4;
      const contrast = Math.sqrt(Math.max(0, (squareSum / count) - ((sum / count) ** 2)));
      if (contrast < 5) return { ok: false, message: "Lighting/contrast is too low. Retake this face slot." };
      return { ok: true, message: "Browser quality check passed; server face detection will run at save." };
    } catch (error) {
      return { ok: false, message: error.message || "The image could not be previewed." };
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  async function validateVoiceFile(file) {
    if (!isAudioFile(file)) return { ok: false, message: "Choose a WAV, WebM, OGG, MP3 or M4A recording." };
    if (file.size <= 0) return { ok: false, message: "The recording file is empty." };
    if (file.size > 20 * 1024 * 1024) return { ok: false, message: "The recording is larger than 20 MB." };
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return { ok: true, deferred: true, message: "Ready; the server will validate this recording." };
    let context;
    try {
      context = new AudioContextClass();
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      if (buffer.duration < 2.5) return { ok: false, message: "Recording is shorter than 2.5 seconds." };
      let squareSum = 0;
      let samples = 0;
      const stride = Math.max(1, Math.floor(buffer.length / 12000));
      for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
        const data = buffer.getChannelData(channel);
        for (let index = 0; index < data.length; index += stride) {
          squareSum += data[index] * data[index];
          samples += 1;
        }
      }
      const rms = Math.sqrt(squareSum / Math.max(samples, 1));
      if (rms < 0.001) return { ok: false, message: "No usable speech detected; this slot is silent or too quiet." };
      return { ok: true, message: "Audio quality check passed; server conversion will run at save." };
    } catch (_) {
      // Some browsers cannot decode WebM/Opus through AudioContext even though
      // the local FFmpeg decoder can. Keep the file and let Flask decide.
      return { ok: true, deferred: true, message: "Browser preview unavailable; the server will validate this recording." };
    } finally {
      if (context && context.close) context.close().catch(() => {});
    }
  }

  async function setDraftFile(kind, index, file, source) {
    if (!file || index < 0 || index >= SLOT_COUNT) return;
    const slots = slotsFor(kind);
    const slot = slots[index];
    const previous = { file: slot.file, status: slot.status, message: slot.message };
    slot.token += 1;
    const token = slot.token;
    slot.file = file;
    slot.status = "validating";
    slot.message = "Checking this sample...";
    slot.source = source;
    renderSlots(kind);
    const result = await (kind === "faces" ? validateFaceFile(file) : validateVoiceFile(file));
    if (slot.token !== token || slot.file !== file) return;
    if (result.ok) {
      slot.status = result.deferred ? "pending" : "valid";
      slot.message = result.message;
    } else if (previous.file && previous.status !== "invalid") {
      // A failed replacement must not destroy a previously valid sample.
      slot.file = previous.file;
      slot.status = previous.status;
      slot.message = `Replacement rejected: ${result.message} Existing sample kept.`;
    } else {
      slot.status = "invalid";
      slot.message = result.message;
    }
    renderSlots(kind);
    updateCounts();
  }

  async function assignFiles(kind, files, targetIndex = null) {
    const list = Array.from(files || []);
    let nextIndex = targetIndex;
    for (const file of list) {
      if (nextIndex === null || nextIndex === undefined) {
        nextIndex = slotsFor(kind).findIndex((slot) => !slot.file || slot.status === "invalid");
      }
      if (nextIndex < 0) break;
      await setDraftFile(kind, nextIndex, file, "upload");
      nextIndex = targetIndex === null || targetIndex === undefined ? null : nextIndex;
    }
    if (list.length > 0 && nextIndex !== null && nextIndex !== undefined && nextIndex >= SLOT_COUNT) {
      setFormError("All four slots are already filled. Replace a slot instead of adding another file.");
    }
    updateCounts();
  }

  function openPicker(kind, index = null) {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = kind === "faces" ? FACE_ACCEPT : VOICE_ACCEPT;
    picker.multiple = index === null;
    picker.addEventListener("change", () => {
      if (picker.files.length) assignFiles(kind, picker.files, index);
    }, { once: true });
    picker.click();
  }

  function button(label, className, handler) {
    const node = document.createElement("button");
    node.type = "button";
    node.className = className || "slot-action";
    node.textContent = label;
    node.addEventListener("click", handler);
    return node;
  }

  function renderSlot(element, kind, index) {
    const slot = slotsFor(kind)[index];
    const labels = labelsFor(kind);
    revokePreview(slot);
    element.replaceChildren();
    element.className = `sample-slot ${slot.file ? "filled" : "empty"} slot-${slot.status}`;
    if (kind === "faces" && index === activeFaceSlot) element.classList.add("slot-selected");
    if (kind === "voices" && index === activeVoiceSlot) element.classList.add("slot-selected");

    const number = document.createElement("b");
    number.textContent = String(index + 1);
    element.appendChild(number);
    const title = document.createElement("strong");
    title.className = "slot-title";
    title.textContent = labels[index];
    element.appendChild(title);

    if (slot.file && kind === "faces") {
      const image = document.createElement("img");
      image.className = "slot-preview";
      image.alt = `${labels[index]} preview`;
      slot.previewUrl = URL.createObjectURL(slot.file);
      image.src = slot.previewUrl;
      element.appendChild(image);
    } else if (slot.file && kind === "voices") {
      const audio = document.createElement("audio");
      audio.className = "slot-audio";
      audio.controls = true;
      audio.preload = "metadata";
      slot.previewUrl = URL.createObjectURL(slot.file);
      audio.src = slot.previewUrl;
      element.appendChild(audio);
    }

    const status = document.createElement("small");
    status.className = "slot-status";
    status.textContent = slot.file ? (slot.message || slot.status) : "Empty - choose capture or upload";
    element.appendChild(status);

    if (slot.file) {
      const fileName = document.createElement("small");
      fileName.className = "slot-name";
      fileName.textContent = slot.file.name;
      element.appendChild(fileName);
    }

    const actions = document.createElement("span");
    actions.className = "slot-actions";
    if (kind === "faces") {
      actions.appendChild(button(slot.file ? "Retake" : "Capture", "slot-action", () => selectFaceSlot(index)));
    } else {
      actions.appendChild(button(slot.file ? "Re-record" : "Record", "slot-action", () => {
        selectVoiceSlot(index);
        recordVoice();
      }));
    }
    actions.appendChild(button(slot.file ? "Replace" : "Upload", "slot-action", () => openPicker(kind, index)));
    if (slot.file) actions.appendChild(button("Remove", "slot-action slot-delete", () => removeSlot(kind, index)));
    element.appendChild(actions);
  }

  function renderSlots(kind) {
    const container = $(kind === "faces" ? "face-slots" : "voice-slots");
    if (!container) return;
    Array.from(container.querySelectorAll(".sample-slot")).forEach((element, index) => renderSlot(element, kind, index));
  }

  function slotReady(slot) {
    return Boolean(slot.file && slot.status !== "invalid" && slot.status !== "validating");
  }

  function updateCounts() {
    const faceReady = draft.faces.filter(slotReady).length;
    const voiceReady = draft.voices.filter(slotReady).length;
    const faceInvalid = draft.faces.filter((slot) => slot.status === "invalid").length;
    const voiceInvalid = draft.voices.filter((slot) => slot.status === "invalid").length;
    $("face-count").textContent = `${faceReady} / ${SLOT_COUNT}`;
    $("voice-count").textContent = `${voiceReady} / ${SLOT_COUNT}`;
    let summary = `Complete ${faceReady} / 4 face samples and ${voiceReady} / 4 voice samples.`;
    if (faceInvalid || voiceInvalid) summary = `Fix ${faceInvalid + voiceInvalid} highlighted sample slot${faceInvalid + voiceInvalid === 1 ? "" : "s"}; other valid samples are preserved.`;
    else if (faceReady >= SLOT_COUNT && voiceReady >= SLOT_COUNT) summary = "All required samples are ready. Review the information and consent, then save.";
    $("save-summary").textContent = summary;
    const info = readInfo();
    const ready = faceReady >= SLOT_COUNT && voiceReady >= SLOT_COUNT && info.display_name && /^[A-Za-z0-9_-]{2,40}$/.test(info.identifier) && info.consent;
    $("save-enroll").disabled = !ready || submitting;
    renderSlots("faces");
    renderSlots("voices");
  }

  function selectFaceSlot(index) {
    activeFaceSlot = index;
    $("face-capture-status").textContent = `Camera target: ${FACE_LABELS[index]}. Enable the camera, then capture this slot.`;
    $("capture-face").textContent = `Capture ${FACE_LABELS[index]}`;
    $("capture-face").disabled = !cameraCapture.stream;
    renderSlots("faces");
  }

  function selectVoiceSlot(index) {
    activeVoiceSlot = index;
    $("voice-record-status").textContent = `Recording target: ${VOICE_LABELS[index]}. Say "BioGuard access" clearly for three seconds.`;
    renderSlots("voices");
  }

  async function enableCamera() {
    try {
      await cameraCapture.startCamera();
      $("enroll-camera-placeholder").classList.add("hidden");
      $("enroll-camera-start").disabled = true;
      $("capture-face").disabled = false;
      $("face-capture-status").textContent = `Camera ready for ${FACE_LABELS[activeFaceSlot]}.`;
    } catch (error) {
      $("enroll-camera-placeholder").querySelector("strong").textContent = error.message || "Camera unavailable";
      $("face-capture-status").textContent = error.message || "Camera unavailable. Upload this slot instead.";
    }
  }

  async function captureFace() {
    const index = activeFaceSlot ?? draft.faces.findIndex((slot) => !slot.file);
    if (!cameraCapture.stream) {
      $("face-capture-status").textContent = "Enable the camera first. Your other slots remain unchanged.";
      return;
    }
    try {
      const file = await cameraCapture.snapshot(`face-${index + 1}-${Date.now()}.jpg`);
      await setDraftFile("faces", index, file, "camera");
      $("face-capture-status").textContent = `${FACE_LABELS[index]} captured. Select another slot or continue editing the form.`;
    } catch (error) {
      $("face-capture-status").textContent = error.message || "Capture failed. Only the selected slot was affected.";
    }
  }

  function stopVoiceTimer() {
    if (voiceTimer) window.clearInterval(voiceTimer);
    voiceTimer = null;
    $("enroll-voice-timer").textContent = "00:00";
  }

  function startVoiceTimer(durationMs) {
    const started = Date.now();
    voiceTimer = window.setInterval(() => {
      const elapsed = Math.min(Date.now() - started, durationMs);
      $("enroll-voice-timer").textContent = `00:${String(Math.floor(elapsed / 1000)).padStart(2, "0")}`;
    }, 100);
  }

  async function recordVoice() {
    const buttonNode = $("record-enroll-voice");
    if (buttonNode.disabled || submitting) return;
    const index = activeVoiceSlot ?? draft.voices.findIndex((slot) => !slot.file);
    const previous = draft.voices[index].file;
    buttonNode.disabled = true;
    $("voice-orb").classList.add("recording");
    $("voice-record-status").textContent = `Recording ${VOICE_LABELS[index]}... say BioGuard access clearly.`;
    startVoiceTimer(3000);
    try {
      const file = await voiceCapture.recordAudio(3000);
      await setDraftFile("voices", index, file, "microphone");
      $("voice-record-status").textContent = `${VOICE_LABELS[index]} recorded. Other voice slots and user information remain unchanged.`;
    } catch (error) {
      $("voice-record-status").textContent = previous
        ? `Replacement failed: ${error.message || "microphone unavailable"}. Existing recording kept.`
        : (error.message || "Microphone unavailable. Upload this slot instead.");
      if (!previous) {
        draft.voices[index].status = "invalid";
        draft.voices[index].message = error.message || "Microphone capture failed.";
        renderSlots("voices");
      }
    } finally {
      stopVoiceTimer();
      buttonNode.disabled = false;
      $("voice-orb").classList.remove("recording");
      updateCounts();
    }
  }

  function removeSlot(kind, index) {
    const slot = slotsFor(kind)[index];
    revokePreview(slot);
    slotsFor(kind)[index] = emptySlot();
    if (kind === "faces") selectFaceSlot(index);
    else selectVoiceSlot(index);
    setFormError("");
    updateCounts();
  }

  function applyServerErrors(errors) {
    if (!errors || typeof errors !== "object") return;
    clearFieldErrors();
    Object.entries(errors).forEach(([key, value]) => {
      const message = typeof value === "string" ? value : (value?.message || "This item needs attention.");
      const match = /^(faces?|voice)\.(\d+)$/.exec(key);
      if (match) {
        const kind = match[1].startsWith("face") ? "faces" : "voices";
        const index = Number(match[2]);
        const slot = slotsFor(kind)[index];
        if (slot) {
          slot.status = "invalid";
          slot.message = message;
          renderSlots(kind);
        }
      } else {
        setFieldError(key === "permissions" ? "permissions" : key, message);
      }
    });
    updateCounts();
  }

  function appendDraft(formData) {
    const info = readInfo();
    formData.set("csrf_token", form.querySelector('[name="csrf_token"]')?.value || "");
    formData.set("replace_identifier", form.querySelector('[name="replace_identifier"]')?.value || "");
    formData.set("display_name", info.display_name);
    formData.set("identifier", info.identifier);
    formData.set("role", info.role);
    if (info.active) formData.set("active", "on");
    info.permissions.forEach((permission) => formData.append("access_permissions", permission));
    if (info.consent) formData.set("consent", "on");
    draft.faces.forEach((slot, index) => formData.append("face_images", slot.file, slot.file.name || `face-${index + 1}.jpg`));
    draft.voices.forEach((slot, index) => formData.append("voice_recordings", slot.file, slot.file.name || `voice-${index + 1}.wav`));
  }

  async function submitEnrollment(event) {
    event.preventDefault();
    if (submitting) return;
    setFormError("");
    setSaveStatus("");
    const infoErrors = validateInfo();
    const invalidSlots = [
      ...draft.faces.map((slot, index) => ({ kind: "faces", slot, index })),
      ...draft.voices.map((slot, index) => ({ kind: "voices", slot, index })),
    ].filter((entry) => !slotReady(entry.slot));
    if (Object.keys(infoErrors).length || invalidSlots.length) {
      if (invalidSlots.length) setFormError("Fix the highlighted fields or sample slots. Valid samples remain available for retry.");
      updateCounts();
      document.querySelector(".input-error, .slot-invalid")?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    const data = new FormData();
    appendDraft(data);
    submitting = true;
    $("save-enroll").disabled = true;
    setSaveStatus("Validating all samples and saving the enrollment atomically...");
    try {
      const response = await fetch(form.dataset.api || "/api/admin/enroll", {
        method: "POST",
        headers: { "X-CSRFToken": form.querySelector('[name="csrf_token"]')?.value || "" },
        body: data,
        credentials: "same-origin",
      });
      let result;
      try {
        result = await response.json();
      } catch (_) {
        result = { ok: false, error: "The server returned an invalid response." };
      }
      if (!response.ok || !result.ok) {
        applyServerErrors(result.errors);
        setFormError(result.error || "Enrollment was not saved. Correct the highlighted item(s) and retry.");
        setSaveStatus("Nothing was saved. Your draft and all valid samples are still here.");
        return;
      }
      setSaveStatus(result.message || "Enrollment saved successfully.", true);
      setFormError("");
      window.setTimeout(() => window.location.assign(result.redirect || "/admin/users"), 450);
    } catch (error) {
      setFormError("The server could not be reached. Nothing was saved; your draft remains available.");
      setSaveStatus(error.message || "Save failed; retry without re-capturing samples.");
    } finally {
      submitting = false;
      updateCounts();
    }
  }

  faceInput.addEventListener("change", () => {
    assignFiles("faces", faceInput.files);
    faceInput.value = "";
  });
  voiceInput.addEventListener("change", () => {
    assignFiles("voices", voiceInput.files);
    voiceInput.value = "";
  });
  $("enroll-camera-start").addEventListener("click", enableCamera);
  $("capture-face").addEventListener("click", captureFace);
  $("record-enroll-voice").addEventListener("click", recordVoice);
  form.addEventListener("submit", submitEnrollment);
  form.querySelectorAll("[data-draft-field], [data-draft-permission]").forEach((node) => {
    node.addEventListener("input", () => { readInfo(); setFormError(""); updateCounts(); });
    node.addEventListener("change", () => { readInfo(); setFormError(""); updateCounts(); });
  });
  window.addEventListener("beforeunload", () => {
    cameraCapture.stopCamera();
    voiceCapture.stopAudio();
    stopVoiceTimer();
    draft.faces.concat(draft.voices).forEach(revokePreview);
  });

  readInfo();
  selectFaceSlot(0);
  selectVoiceSlot(0);
  updateCounts();
})();
