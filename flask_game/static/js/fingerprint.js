(function () {
  "use strict";

  const form = document.getElementById("fingerprint-form");
  if (!form) return;

  const input = document.getElementById("fingerprint_image");
  const dropzone = document.getElementById("fingerprint-dropzone");
  const previewFrame = document.getElementById("fingerprint-preview-frame");
  const preview = document.getElementById("fingerprint-preview");
  const previewPlaceholder = document.getElementById("fingerprint-preview-placeholder");
  const fileName = document.getElementById("fingerprint-file-name");
  const fileStatus = document.getElementById("fingerprint-file-status");
  const submit = document.getElementById("fingerprint-submit");
  const status = document.getElementById("fingerprint-status");
  const statusText = document.getElementById("fingerprint-status-text");
  const result = document.getElementById("fingerprint-result");
  const resultDecision = document.getElementById("fingerprint-decision");
  const resultReason = document.getElementById("fingerprint-reason");
  const resultBadge = document.getElementById("fingerprint-result-badge");
  const resultError = document.getElementById("fingerprint-error");
  const resultErrorCode = document.getElementById("fingerprint-error-code");
  const resultErrorDetail = document.getElementById("fingerprint-error-detail");
  const visuals = document.getElementById("feature-visuals");
  const score = document.getElementById("fingerprint-score");
  const threshold = document.getElementById("fingerprint-threshold");
  const model = document.getElementById("fingerprint-model");
  const gallery = document.getElementById("fingerprint-gallery");
  const footnote = document.getElementById("fingerprint-result-footnote");
  const csrf = form.querySelector('input[name="csrf_token"]') || document.querySelector('meta[name="csrf-token"]');
  const supported = new Set(["png", "jpg", "jpeg", "bmp"]);
  let previewUrl = "";
  let selectedFile = null;

  function extension(file) {
    const name = String(file && file.name ? file.name : "");
    const pieces = name.toLowerCase().split(".");
    return pieces.length > 1 ? pieces.pop() : "";
  }

  function validateFile(file) {
    if (!file) return { code: "NO_FILE", detail: "Choose a fingerprint image first." };
    if (!supported.has(extension(file))) {
      return { code: "INVALID_EXTENSION", detail: "Use a PNG, JPG, JPEG, or BMP fingerprint image." };
    }
    if (file.size === 0) return { code: "EMPTY_UPLOAD", detail: "The selected file is empty." };
    if (file.size > 64 * 1024 * 1024) {
      return { code: "UPLOAD_TOO_LARGE", detail: "The selected image is larger than the 64 MB local upload limit." };
    }
    return null;
  }

  function setStatus(message, kind) {
    status.classList.remove("is-working", "is-error", "is-success");
    if (kind) status.classList.add("is-" + kind);
    statusText.textContent = message;
  }

  function showError(code, detail) {
    result.hidden = false;
    result.classList.remove("is-match");
    result.classList.add("is-deny");
    resultDecision.textContent = "NO MATCH";
    resultBadge.textContent = "NO MATCH";
    resultReason.textContent = detail || "The fingerprint scan was rejected.";
    resultError.hidden = false;
    resultErrorCode.textContent = code || "FINGERPRINT_PROCESSING_FAILED";
    resultErrorDetail.textContent = detail || "The scan could not be completed.";
    score.textContent = "—";
    threshold.textContent = "—";
    model.textContent = "—";
    gallery.textContent = "—";
    footnote.textContent = "Feature vector: — dimensions · Pair vector: — dimensions · Compared with: — references";
    visuals.hidden = true;
    setStatus((code || "ERROR") + " — " + (detail || "The scan could not be completed."), "error");
  }

  function hideResult() {
    result.hidden = true;
    result.classList.remove("is-match", "is-deny");
    resultError.hidden = true;
    visuals.hidden = true;
  }

  function textOrDash(value) {
    return value === undefined || value === null || value === "" ? "—" : String(value);
  }

  function numberOrDash(value) {
    return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
  }

  function showResult(payload) {
    const matched = payload.match === true;
    result.hidden = false;
    result.classList.toggle("is-match", matched);
    result.classList.toggle("is-deny", !matched);
    resultDecision.textContent = textOrDash(payload.decision || (matched ? "MATCH" : "NO MATCH"));
    resultBadge.textContent = matched ? "MATCH" : "NO MATCH";
    resultReason.textContent = textOrDash(payload.reason || payload.detail);
    resultError.hidden = !payload.error_code;
    resultErrorCode.textContent = textOrDash(payload.error_code);
    resultErrorDetail.textContent = textOrDash(payload.detail || payload.reason);
    score.textContent = numberOrDash(payload.score);
    threshold.textContent = numberOrDash(payload.threshold);
    model.textContent = textOrDash(payload.model_name);
    gallery.textContent = textOrDash(payload.gallery_count);
    footnote.textContent = "Feature vector: " + textOrDash(payload.feature_dim) +
      " dimensions · Pair vector: " + textOrDash(payload.pair_feature_dim) +
      " dimensions · Compared with: " + textOrDash(payload.compared_with || payload.gallery_count) + " references";

    const imageMap = payload.visualizations || {};
    const imageIds = {
      original: "fingerprint-original",
      grayscale: "fingerprint-grayscale",
      sobel: "fingerprint-sobel",
      lbp: "fingerprint-lbp"
    };
    let hasVisuals = false;
    Object.keys(imageIds).forEach(function (key) {
      const image = document.getElementById(imageIds[key]);
      if (image && imageMap[key]) {
        image.src = imageMap[key];
        hasVisuals = true;
      }
    });
    visuals.hidden = !hasVisuals;
    setStatus(matched ? "Real model decision: MATCH." : "Real model decision: NO MATCH.", matched ? "success" : "error");
  }

  function showPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewFrame.hidden = false;
    previewPlaceholder.hidden = true;
    fileName.textContent = file.name;
    fileStatus.textContent = "Ready to run · " + Math.ceil(file.size / 1024) + " KB";
    dropzone.classList.add("is-ready");
  }

  function resetUpload(focus) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = "";
    selectedFile = null;
    input.value = "";
    preview.removeAttribute("src");
    previewFrame.hidden = true;
    previewPlaceholder.hidden = false;
    dropzone.classList.remove("is-ready");
    hideResult();
    setStatus("Waiting for an image.", null);
    if (focus) input.focus();
  }

  input.addEventListener("change", function () {
    const file = input.files && input.files[0];
    const error = validateFile(file);
    hideResult();
    if (error) {
      selectedFile = null;
      showError(error.code, error.detail);
      return;
    }
    selectedFile = file;
    showPreview(file);
    setStatus("Image ready. Run the offline match when you are ready.", null);
  });

  [document.getElementById("fingerprint-new-upload"), document.getElementById("fingerprint-reset-top")]
    .filter(Boolean)
    .forEach(function (button) { button.addEventListener("click", function () { resetUpload(true); }); });

  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    const file = selectedFile || (input.files && input.files[0]);
    const validationError = validateFile(file);
    if (validationError) {
      showError(validationError.code, validationError.detail);
      return;
    }
    selectedFile = file;
    submit.disabled = true;
    submit.dataset.originalLabel = submit.innerHTML;
    submit.innerHTML = "Processing <span class=\"button-spinner\" aria-hidden=\"true\"></span>";
    setStatus("Decoding image, extracting Sobel/LBP features, and verifying…", "working");
    result.hidden = true;
    const headers = { "Accept": "application/json" };
    if (csrf && csrf.value) headers["X-CSRFToken"] = csrf.value;
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        credentials: "same-origin",
        headers: headers
      });
      let payload;
      try {
        payload = await response.json();
      } catch (_jsonError) {
        payload = { ok: false, error_code: "SERVER_ERROR", detail: "The server returned an unreadable response." };
      }
      if (!response.ok || payload.ok === false) {
        showError(payload.error_code || "SERVER_ERROR", payload.detail || payload.reason || "Fingerprint verification failed.");
      } else {
        showResult(payload);
      }
    } catch (_networkError) {
      showError("SERVER_ERROR", "The fingerprint service could not be reached. Your selected image is still available to retry.");
    } finally {
      submit.disabled = false;
      submit.innerHTML = submit.dataset.originalLabel || "Run offline match";
    }
  });

  setStatus("Waiting for an image.", null);
}());
