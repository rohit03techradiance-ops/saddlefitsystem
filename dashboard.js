(function () {
  const config = window.__SADDLEFIT_CONFIG__ || {};
  const uploadMode = config.uploadMode || "multipart";
  const storageAccess = config.storageAccess || "private";
  const maxVideoBytes = Number(config.maxVideoBytes || 200 * 1024 * 1024);
  const maxVideoLabel = config.maxVideoLabel || Math.max(1, Math.round(maxVideoBytes / (1024 * 1024))) + " MB";
  const blobUploadUrl = config.blobUploadUrl || "/api/blob-upload";
  const analysisApiUrl = config.analysisApiUrl || "/api/analyze";
  const compareApiUrl = config.compareApiUrl || "/api/compare";
  const blobClientImportUrl = config.blobClientImportUrl || "https://esm.sh/@vercel/blob@2.6.1/client";
  const allowedExtensions = [".mp4", ".mov"];
  const allowedMimeTypes = ["video/mp4", "video/quicktime"];

  const uploadStepsSingle = [
    { title: "Preparing Upload", subtitle: "Validating the clip and securing the upload." },
    { title: "Uploading Video", subtitle: "Sending the ride directly to blob storage." },
    { title: "Upload Complete", subtitle: "Starting the analysis request." },
  ];

  const uploadStepsCompare = [
    { title: "Preparing Upload", subtitle: "Validating both clips and securing the uploads." },
    { title: "Uploading Video A", subtitle: "Sending the first ride directly to blob storage." },
    { title: "Uploading Video B", subtitle: "Sending the second ride directly to blob storage." },
    { title: "Upload Complete", subtitle: "Starting the comparison request." },
  ];

  const analysisStepsSingle = [
    { title: "Starting Analysis", subtitle: "Sending a small JSON request to FastAPI." },
    { title: "Downloading for Processing", subtitle: "Streaming the video into /tmp/saddlefitsystem." },
    { title: "Analyzing Rider", subtitle: "Evaluating posture, balance, and symmetry." },
    { title: "Analyzing Horse", subtitle: "Measuring movement, rhythm, and consistency." },
    { title: "Analyzing Saddle", subtitle: "Checking stability, alignment, and clearance." },
    { title: "Calculating Metrics", subtitle: "Building scores and recommendations." },
    { title: "Generating Report", subtitle: "Rendering the HTML and PDF outputs." },
    { title: "Analysis Complete", subtitle: "Opening the finished report." },
  ];

  const analysisStepsCompare = [
    { title: "Starting Comparison", subtitle: "Sending both blob references to FastAPI." },
    { title: "Downloading Ride A", subtitle: "Streaming the first video into /tmp/saddlefitsystem." },
    { title: "Analyzing Ride A", subtitle: "Scoring the first ride." },
    { title: "Downloading Ride B", subtitle: "Streaming the second video into /tmp/saddlefitsystem." },
    { title: "Analyzing Ride B", subtitle: "Scoring the second ride." },
    { title: "Comparing Metrics", subtitle: "Calculating side-by-side differences." },
    { title: "Generating Report", subtitle: "Rendering the comparison HTML and PDF outputs." },
    { title: "Comparison Complete", subtitle: "Opening the finished report." },
  ];

  const state = {
    timer: null,
    directUploader: null,
  };

  function el(id) {
    return document.getElementById(id);
  }

  function openHorseGuide() {
    window.open("/horse-scheme-guide", "_blank", "noopener");
  }

  function openDisciplineGuide() {
    window.open("/discipline-guide", "_blank", "noopener");
  }

  function openSchemeGuide() {
    openHorseGuide();
  }

  function closeSchemeGuide() {
    const modal = el("schemeModal");
    if (modal) modal.style.display = "none";
  }

  function syncLegacySelection(isCompare) {
    const profileSelect = el(isCompare ? "horseProfileCompare" : "horseProfileSelect");
    const disciplineSelect = el(isCompare ? "disciplineCompare" : "disciplineSelect");
    const hiddenScheme = el(isCompare ? "horseSchemeCompare" : "horseScheme");
    if (profileSelect && hiddenScheme) hiddenScheme.value = profileSelect.value || "high_wither";
    if (disciplineSelect && !disciplineSelect.value) disciplineSelect.value = "general_riding";
  }

  function quickProfile(val, isCompare) {
    const select = el(isCompare ? "horseProfileCompare" : "horseProfileSelect");
    if (select) select.value = val;
    syncLegacySelection(!!isCompare);
  }

  function quickDiscipline(val, isCompare) {
    const select = el(isCompare ? "disciplineCompare" : "disciplineSelect");
    if (select) select.value = val;
    syncLegacySelection(!!isCompare);
  }

  function quickScheme(val) {
    const alias = {
      trail: "trail_riding",
      dressage: "dressage",
      racing: "racing_gallop",
      show_jumping: "show_jumping",
    };
    const mapped = alias[val] || val;
    quickDiscipline(mapped, false);
    quickDiscipline(mapped, true);
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    let value = bytes;
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }
    return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
  }

  function sanitizeFileName(name) {
    const parts = String(name || "video.mp4").split(/[\\/]/);
    const base = parts[parts.length - 1] || "video.mp4";
    const dotIndex = base.lastIndexOf(".");
    const stem = dotIndex > 0 ? base.slice(0, dotIndex) : base;
    const ext = dotIndex > 0 ? base.slice(dotIndex).toLowerCase() : ".mp4";
    const safeStem = stem.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "video";
    const safeExt = allowedExtensions.includes(ext) ? ext : ".mp4";
    return `${safeStem.slice(0, 80)}${safeExt}`;
  }

  function inferMimeType(file) {
    const explicit = String(file && file.type ? file.type : "").split(";")[0].toLowerCase();
    if (allowedMimeTypes.includes(explicit)) return explicit;
    const ext = String(file && file.name ? file.name : "").toLowerCase().split(".").pop();
    if (ext === "mov") return "video/quicktime";
    return "video/mp4";
  }

  function isSupportedVideoFile(file) {
    if (!file) return false;
    const name = String(file.name || "").toLowerCase();
    const ext = name.includes(".") ? `.${name.split(".").pop()}` : "";
    const mime = String(file.type || "").toLowerCase().split(";")[0];
    return allowedExtensions.includes(ext) || allowedMimeTypes.includes(mime);
  }

  function buildBlobPath(scope, file, slot) {
    const safeName = sanitizeFileName(file.name);
    const id = window.crypto && typeof window.crypto.randomUUID === "function"
      ? window.crypto.randomUUID().replace(/-/g, "")
      : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
    const parts = ["videos", scope];
    if (slot) parts.push(slot);
    parts.push(`${id}-${safeName}`);
    return parts.join("/");
  }

  function validateVideoFile(file, label) {
    if (!file) {
      throw new Error(`Please choose ${label}.`);
    }
    if (file.size > maxVideoBytes) {
      throw new Error(`Selected file exceeds the ${maxVideoLabel} upload limit.`);
    }
    if (!isSupportedVideoFile(file)) {
      throw new Error("Only MP4 and MOV videos are supported.");
    }
  }

  function getOverlayNodes() {
    return {
      overlay: el("progressOverlay"),
      stepsNode: el("progressSteps"),
      titleNode: el("progressTitle"),
      subtitleNode: el("progressSubtitle"),
      percentNode: el("progressPercent"),
      fillNode: el("progressFill"),
    };
  }

  function stopOverlayTimer() {
    if (state.timer) {
      window.clearInterval(state.timer);
      state.timer = null;
    }
  }

  function hideOverlay() {
    const nodes = getOverlayNodes();
    if (nodes.overlay) {
      nodes.overlay.style.display = "none";
    }
    document.body.style.overflow = "";
  }

  function renderOverlaySteps(steps, activeIndex) {
    const nodes = getOverlayNodes();
    if (!nodes.stepsNode) return;
    nodes.stepsNode.innerHTML = steps
      .map(
        (step, index) => `
          <div class="progress-step${index === activeIndex ? " active" : ""}" data-progress-step="${index}">
            <b>${step.title}</b>
            <div>${step.subtitle}</div>
          </div>
        `,
      )
      .join("");
  }

  function renderOverlay({ title, subtitle, percentText, percentValue, indeterminate, steps, activeIndex, error }) {
    const nodes = getOverlayNodes();
    if (!nodes.overlay || !nodes.stepsNode || !nodes.titleNode || !nodes.subtitleNode || !nodes.percentNode || !nodes.fillNode) {
      return;
    }
    nodes.overlay.style.display = "flex";
    document.body.style.overflow = "hidden";
    nodes.titleNode.textContent = title;
    nodes.subtitleNode.textContent = subtitle;
    nodes.percentNode.textContent = percentText;
    nodes.fillNode.classList.toggle("indeterminate", !!indeterminate);
    nodes.fillNode.style.width = indeterminate ? "58%" : `${Math.max(0, Math.min(100, Number(percentValue) || 0))}%`;
    nodes.fillNode.style.background = error
      ? "linear-gradient(135deg, #fb7185, #ef4444)"
      : "linear-gradient(135deg, #5be286, #2dd4bf)";
    renderOverlaySteps(steps || [], Number.isFinite(activeIndex) ? activeIndex : 0);
  }

  function showError(message) {
    stopOverlayTimer();
    renderOverlay({
      title: "Upload failed",
      subtitle: message || "An unexpected error occurred.",
      percentText: "Error",
      percentValue: 100,
      indeterminate: false,
      steps: [{ title: "Error", subtitle: message || "Please try again." }],
      activeIndex: 0,
      error: true,
    });
    const nodes = getOverlayNodes();
    if (nodes.overlay) {
      nodes.overlay.addEventListener(
        "click",
        (event) => {
          if (event.target === nodes.overlay) hideOverlay();
        },
        { once: true },
      );
    }
  }

  function setFormBusy(form, busy) {
    if (!form) return;
    const button = form.querySelector('button[type="submit"]');
    if (button) button.disabled = !!busy;
  }

  function currentProfile(isCompare) {
    const select = el(isCompare ? "horseProfileCompare" : "horseProfileSelect");
    return select && select.value ? select.value : "high_wither";
  }

  function currentDiscipline(isCompare) {
    const select = el(isCompare ? "disciplineCompare" : "disciplineSelect");
    return select && select.value ? select.value : "general_riding";
  }

  function currentSaddleType(isCompare) {
    const select = el(isCompare ? "saddleTypeCompare" : "saddleType");
    if (select && select.value) return select.value;
    const form = el(isCompare ? "compareForm" : "uploadForm");
    const fallback = form ? form.querySelector('select[name="saddle_type_compare"], select[name="saddle_type"]') : null;
    return fallback && fallback.value ? fallback.value : "english";
  }

  function getFileFromForm(form, name) {
    const input = form ? form.querySelector(`input[name="${name}"]`) : null;
    return input && input.files && input.files.length ? input.files[0] : null;
  }

  function buildAnalysisPayload(blob, file, isCompare) {
    return {
      video_url: blob.url,
      storage_key: blob.pathname,
      original_filename: file.name,
      storage_provider: "vercel_blob",
      horse_profile: currentProfile(isCompare),
      saddle_type: currentSaddleType(isCompare),
      discipline: currentDiscipline(isCompare),
    };
  }

  function buildComparisonPayload(blobA, blobB, fileA, fileB) {
    return {
      video_a_url: blobA.url,
      video_b_url: blobB.url,
      video_a_key: blobA.pathname,
      video_b_key: blobB.pathname,
      video_a_filename: fileA.name,
      video_b_filename: fileB.name,
      storage_provider: "vercel_blob",
      horse_profile: currentProfile(true),
      saddle_type: currentSaddleType(true),
      discipline: currentDiscipline(true),
    };
  }

  function parseResponsePayload(text) {
    if (!text) return {};
    try {
      return JSON.parse(text);
    } catch (_error) {
      return { raw: text };
    }
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    });
    const body = await response.text();
    const data = parseResponsePayload(body);
    if (!response.ok) {
      throw new Error(data.error || data.detail || data.message || `Request failed with status ${response.status}.`);
    }
    return data;
  }

  function redirectToReport(payload) {
    const target = payload && (payload.report_url || payload.pdf_url || (payload.analysis_id ? `/report/${payload.analysis_id}` : "") || (payload.comparison_id ? `/compare_report/${payload.comparison_id}` : ""));
    if (!target) {
      throw new Error("The report URL was missing from the response.");
    }
    window.location.assign(target);
  }

  function startAnalysisSequence(steps, comparisonMode) {
    stopOverlayTimer();
    let activeIndex = 0;
    renderOverlay({
      title: steps[0].title,
      subtitle: steps[0].subtitle,
      percentText: "Working",
      percentValue: 60,
      indeterminate: true,
      steps,
      activeIndex,
    });
    state.timer = window.setInterval(() => {
      if (activeIndex < steps.length - 1) {
        activeIndex += 1;
        renderOverlay({
          title: steps[activeIndex].title,
          subtitle: steps[activeIndex].subtitle,
          percentText: "Working",
          percentValue: 60,
          indeterminate: true,
          steps,
          activeIndex,
        });
        return;
      }
      stopOverlayTimer();
    }, 1100);
  }

  function renderUploadState(steps, activeIndex, title, subtitle, percent, indeterminate) {
    renderOverlay({
      title,
      subtitle,
      percentText: indeterminate ? "Working" : `${Math.max(0, Math.min(100, Math.round(percent || 0)))}%`,
      percentValue: percent || 0,
      indeterminate: !!indeterminate,
      steps,
      activeIndex,
    });
  }

  async function getDirectUploader() {
    if (state.directUploader) return state.directUploader;
    try {
      state.directUploader = await import(blobClientImportUrl);
      return state.directUploader;
    } catch (_error) {
      throw new Error("Direct upload could not load the blob client module. Please retry or use local development mode.");
    }
  }

  async function uploadVideo(file, scope, slot, onProgress) {
    const uploader = await getDirectUploader();
    const pathname = buildBlobPath(scope, file, slot);
    return uploader.upload(pathname, file, {
      access: storageAccess,
      handleUploadUrl: blobUploadUrl,
      multipart: true,
      contentType: inferMimeType(file),
      onUploadProgress: onProgress,
    });
  }

  async function handleDirectAnalysis(form, isCompare) {
    const steps = isCompare ? uploadStepsCompare : uploadStepsSingle;
    const analysisSteps = isCompare ? analysisStepsCompare : analysisStepsSingle;
    const file = getFileFromForm(form, isCompare ? "video_a" : "video");
    validateVideoFile(file, isCompare ? "Video A" : "a video");
    setFormBusy(form, true);
    try {
      stopOverlayTimer();
      renderUploadState(
        steps,
        0,
        steps[0].title,
        steps[0].subtitle,
        0,
        true,
      );

      let currentPercent = 0;
      const blob = await uploadVideo(file, isCompare ? "comparison" : "analysis", "", (event) => {
        const percent = typeof event.percentage === "number"
          ? event.percentage
          : event.total
            ? (event.loaded / event.total) * 100
            : 0;
        currentPercent = Math.max(currentPercent, percent || 0);
        renderUploadState(
          steps,
          currentPercent >= 100 ? steps.length - 1 : 1,
          currentPercent >= 100 ? steps[steps.length - 1].title : steps[1].title,
          currentPercent >= 100 ? steps[steps.length - 1].subtitle : steps[1].subtitle,
          currentPercent,
          false,
        );
      });

      renderUploadState(
        steps,
        steps.length - 1,
        steps[steps.length - 1].title,
        steps[steps.length - 1].subtitle,
        100,
        false,
      );

      startAnalysisSequence(analysisSteps, false);
      const response = await postJson(analysisApiUrl, buildAnalysisPayload(blob, file, isCompare));
      stopOverlayTimer();
      renderOverlay({
        title: "Analysis Complete",
        subtitle: "Opening the finished report.",
        percentText: "Done",
        percentValue: 100,
        indeterminate: false,
        steps: analysisSteps,
        activeIndex: analysisSteps.length - 1,
      });
      window.setTimeout(() => redirectToReport(response), 250);
    } catch (error) {
      showError(error && error.message ? error.message : String(error));
      setFormBusy(form, false);
    }
  }

  async function handleDirectComparison(form) {
    const uploadSteps = uploadStepsCompare;
    const analysisSteps = analysisStepsCompare;
    const fileA = getFileFromForm(form, "video_a");
    const fileB = getFileFromForm(form, "video_b");
    validateVideoFile(fileA, "Video A");
    validateVideoFile(fileB, "Video B");
    setFormBusy(form, true);
    try {
      stopOverlayTimer();
      renderUploadState(uploadSteps, 0, uploadSteps[0].title, uploadSteps[0].subtitle, 0, true);

      let percentA = 0;
      const blobA = await uploadVideo(fileA, "comparison", "a", (event) => {
        const percent = typeof event.percentage === "number"
          ? event.percentage
          : event.total
            ? (event.loaded / event.total) * 100
            : 0;
        percentA = Math.max(percentA, percent || 0);
        renderUploadState(
          uploadSteps,
          1,
          uploadSteps[1].title,
          `${uploadSteps[1].subtitle} (${fileA.name})`,
          percentA,
          false,
        );
      });

      let percentB = 0;
      const blobB = await uploadVideo(fileB, "comparison", "b", (event) => {
        const percent = typeof event.percentage === "number"
          ? event.percentage
          : event.total
            ? (event.loaded / event.total) * 100
            : 0;
        percentB = Math.max(percentB, percent || 0);
        renderUploadState(
          uploadSteps,
          2,
          uploadSteps[2].title,
          `${uploadSteps[2].subtitle} (${fileB.name})`,
          percentB,
          false,
        );
      });

      renderUploadState(
        uploadSteps,
        uploadSteps.length - 1,
        uploadSteps[uploadSteps.length - 1].title,
        uploadSteps[uploadSteps.length - 1].subtitle,
        100,
        false,
      );

      startAnalysisSequence(analysisSteps, true);
      const response = await postJson(compareApiUrl, buildComparisonPayload(blobA, blobB, fileA, fileB));
      stopOverlayTimer();
      renderOverlay({
        title: "Comparison Complete",
        subtitle: "Opening the finished report.",
        percentText: "Done",
        percentValue: 100,
        indeterminate: false,
        steps: analysisSteps,
        activeIndex: analysisSteps.length - 1,
      });
      window.setTimeout(() => redirectToReport(response), 250);
    } catch (error) {
      showError(error && error.message ? error.message : String(error));
      setFormBusy(form, false);
    }
  }

  function showLegacyUploadOverlay(isCompare) {
    const steps = isCompare ? uploadStepsCompare : uploadStepsSingle;
    stopOverlayTimer();
    renderOverlay({
      title: steps[0].title,
      subtitle: steps[0].subtitle + " Local multipart upload in progress.",
      percentText: "Working",
      percentValue: 60,
      indeterminate: true,
      steps,
      activeIndex: 0,
    });
  }

  function prepareForm(formId, isCompare) {
    const form = el(formId);
    if (!form) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      syncLegacySelection(!!isCompare);
      if (uploadMode === "blob") {
        if (isCompare) {
          await handleDirectComparison(form);
        } else {
          await handleDirectAnalysis(form, false);
        }
        return;
      }
      showLegacyUploadOverlay(!!isCompare);
      window.setTimeout(() => form.submit(), 120);
    });
  }

  function wireFormChanges() {
    const profileMain = el("horseProfileSelect");
    const disciplineMain = el("disciplineSelect");
    const profileCompare = el("horseProfileCompare");
    const disciplineCompare = el("disciplineCompare");
    [profileMain, disciplineMain, profileCompare, disciplineCompare].forEach((node) => {
      if (!node) return;
      node.addEventListener("change", () => syncLegacySelection(node.id.includes("Compare")));
    });
  }

  window.openHorseGuide = openHorseGuide;
  window.openDisciplineGuide = openDisciplineGuide;
  window.openSchemeGuide = openSchemeGuide;
  window.closeSchemeGuide = closeSchemeGuide;
  window.syncLegacySelection = syncLegacySelection;
  window.quickProfile = quickProfile;
  window.quickDiscipline = quickDiscipline;
  window.quickScheme = quickScheme;

  document.addEventListener("DOMContentLoaded", () => {
    syncLegacySelection(false);
    syncLegacySelection(true);
    prepareForm("uploadForm", false);
    prepareForm("compareForm", true);
    wireFormChanges();
  });
})();
