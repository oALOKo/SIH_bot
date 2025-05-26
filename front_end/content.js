// content.js
(async () => {
  // ─── 0) Bail out on your redirect target ────────────────────────────────
  const REDIRECT_URL = "https://www.youtube.com/watch?v=P4b7muX5S_g";
  if (window.location.href.startsWith(REDIRECT_URL)) {
    console.log("👾 [liveness] On redirect page, skipping liveness check");
    return;
  }

  console.log("👾 [liveness] Script start");

  // 1) Request camera
  console.log("👾 [liveness] Calling getUserMedia…");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    console.log("👾 [liveness] Camera stream obtained");
  } catch (err) {
    console.error("👾 [liveness] getUserMedia failed:", err);
    // Treat camera failure as spoof
    return window.location.replace(REDIRECT_URL);
  }

  // 2) Attach video and play
  const video = document.createElement("video");
  video.style.display = "none";
  document.body.appendChild(video);
  video.srcObject = stream;
  try {
    await video.play();
    console.log("👾 [liveness] video.play() succeeded");
  } catch (err) {
    console.error("👾 [liveness] video.play() failed:", err);
    stream.getTracks().forEach(t => t.stop());
    return;
  }

  // 3) Warm-up delay
  await new Promise(res => setTimeout(res, 300));
  console.log("👾 [liveness] Warm-up delay complete");

  // 4) Prepare canvas
  const canvas = document.createElement("canvas");
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");

  // 5) Capture a frame
  console.log("👾 [liveness] Capturing frame…");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // 6) Stop camera
  stream.getTracks().forEach(t => t.stop());
  console.log("👾 [liveness] Camera stopped");

  // 7) Convert to blob
  const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg"));
  if (!blob) {
    console.error("👾 [liveness] Failed to create blob");
    return;
  }
  console.log("👾 [liveness] Blob created");

  // 8) Send to backend
  const form = new FormData();
  form.append("frame", blob, "frame.jpg");
  console.log("👾 [liveness] Sending frame to /check_liveness");
  let data;
  try {
    const resp = await fetch("http://127.0.0.1:5000/check_liveness", {
      method: "POST",
      body: form
    });
    data = await resp.json();
    console.log("👾 [liveness] Backend status:", data.status);
  } catch (err) {
    console.error("👾 [liveness] Fetch failed:", err);
    return;
  }

  // 9) Redirect on Attack
  if (data.status === "Attack") {
    console.warn("👾 [liveness] Attack detected—redirecting");
    return window.location.replace(REDIRECT_URL);
  }

  // 10) Otherwise Real
  console.log("👾 [liveness] Real—allowing page");
})();
