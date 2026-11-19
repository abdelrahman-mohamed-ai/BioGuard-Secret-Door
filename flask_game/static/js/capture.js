/* Browser capture helpers. Media never leaves the browser except when the
   page explicitly posts a verification or enrollment request to this origin. */
(function () {
  "use strict";

  class BioGuardCapture {
    constructor(video, canvas) {
      this.video = video;
      this.canvas = canvas;
      this.stream = null;
      this.recorder = null;
      this.recordedChunks = [];
      this.recording = false;
    }

    async startCamera() {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("This browser does not expose camera permissions.");
      }
      this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
      }
      return this.stream;
    }

    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
      }
      if (this.video) this.video.srcObject = null;
    }

    snapshot(filename = "face-capture.jpg") {
      if (!this.video || !this.video.videoWidth) throw new Error("Camera is not ready yet.");
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
      const context = this.canvas.getContext("2d");
      context.save();
      context.translate(this.canvas.width, 0);
      context.scale(-1, 1);
      context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
      context.restore();
      return new Promise((resolve) => this.canvas.toBlob((blob) => resolve(new File([blob], filename, { type: "image/jpeg" })), "image/jpeg", .92));
    }

    async recordAudio(milliseconds = 3000, mimePreferences = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"]) {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        const error = new Error("This browser does not expose microphone permissions.");
        error.code = "AUDIO_CAPTURE_FAILED";
        throw error;
      }
      if (!window.MediaRecorder) {
        const error = new Error("This browser does not support microphone recording.");
        error.code = "AUDIO_CAPTURE_FAILED";
        throw error;
      }
      let audioStream;
      try {
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 }, video: false });
      } catch (error) {
        const captureError = new Error(error?.message || "Microphone permission was not granted.");
        captureError.code = "AUDIO_CAPTURE_FAILED";
        throw captureError;
      }
      const mimeType = mimePreferences.find((type) => MediaRecorder.isTypeSupported(type)) || "";
      this.recordedChunks = [];
      try {
        this.recorder = new MediaRecorder(audioStream, mimeType ? { mimeType } : undefined);
      } catch (error) {
        audioStream.getTracks().forEach((track) => track.stop());
        const recorderError = new Error(`Microphone recorder could not start: ${error.message || error}`);
        recorderError.code = "AUDIO_CAPTURE_FAILED";
        throw recorderError;
      }
      this.recording = true;
      const result = new Promise((resolve, reject) => {
        let settled = false;
        this.recorder.ondataavailable = (event) => { if (event.data && event.data.size) this.recordedChunks.push(event.data); };
        this.recorder.onerror = () => {
          if (settled) return;
          settled = true;
          const error = new Error("Microphone capture failed.");
          error.code = "AUDIO_CAPTURE_FAILED";
          reject(error);
        };
        this.recorder.onstop = () => {
          audioStream.getTracks().forEach((track) => track.stop());
          if (settled) return;
          const type = this.recorder.mimeType || mimeType || "audio/webm";
          const blob = new Blob(this.recordedChunks, { type });
          if (!blob.size) {
            settled = true;
            const error = new Error("The recorder returned an empty audio file.");
            error.code = "AUDIO_UPLOAD_FAILED";
            reject(error);
            return;
          }
          const baseType = type.split(";", 1)[0].toLowerCase();
          const extension = baseType.includes("ogg") ? "ogg" : baseType.includes("mp4") ? "m4a" : "webm";
          settled = true;
          resolve(new File([blob], `voice-${Date.now()}.${extension}`, { type }));
        };
      });
      try {
        // A timeslice makes browsers emit data before stop(), avoiding empty
        // blobs on implementations that delay the final dataavailable event.
        this.recorder.start(250);
        window.setTimeout(() => { if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop(); }, milliseconds);
        return await result;
      } catch (error) {
        audioStream.getTracks().forEach((track) => track.stop());
        if (error?.code) throw error;
        const captureError = new Error(error?.message || "Microphone capture failed.");
        captureError.code = "AUDIO_CAPTURE_FAILED";
        throw captureError;
      } finally {
        this.recording = false;
      }
    }

    stopAudio() {
      if (this.recorder && this.recorder.state !== "inactive") this.recorder.stop();
      this.recording = false;
    }
  }

  window.BioGuardCapture = BioGuardCapture;
})();
