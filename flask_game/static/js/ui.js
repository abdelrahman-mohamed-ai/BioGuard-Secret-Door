(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    const fullscreen = document.getElementById("fullscreen-toggle");
    if (fullscreen) {
      const updateFullscreenLabel = function () {
        const active = Boolean(document.fullscreenElement);
        fullscreen.title = active ? "Exit fullscreen" : "Enter fullscreen";
        fullscreen.setAttribute("aria-label", active ? "Exit fullscreen" : "Toggle fullscreen");
      };
      fullscreen.addEventListener("click", async function () {
        try {
          if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
          else await document.exitFullscreen();
        } catch (_) {
          fullscreen.title = "Fullscreen is not available in this browser";
        }
      });
      document.addEventListener("fullscreenchange", updateFullscreenLabel);
      updateFullscreenLabel();
    }
    document.querySelectorAll("[data-back]").forEach((button) => button.addEventListener("click", () => {
      const sameOrigin = document.referrer && document.referrer.startsWith(window.location.origin);
      if (sameOrigin && window.history.length > 1) window.history.back();
      else window.location.href = "/";
    }));
    document.querySelectorAll("[data-restart]").forEach((button) => button.addEventListener("click", () => window.location.reload()));
  });
})();
