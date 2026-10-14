(() => {
  "use strict";

  const canvas = document.querySelector("#sim-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) return;

  const ctx = canvas.getContext("2d");
  const section = document.querySelector("#drive");
  const startOverlay = document.querySelector("#sim-start");
  const startTitle = startOverlay?.querySelector("strong");
  const startCopy = startOverlay?.querySelector("span");
  const connectionLabel = document.querySelector("#sim-connection");
  const verifier = document.querySelector(".verifier");
  const status = document.querySelector("#verifier-status");
  const speedOutput = document.querySelector("#sim-speed");
  const distanceOutput = document.querySelector("#line-distance");
  const stopOutput = document.querySelector("#stop-state");
  const liveRegion = document.querySelector("#sim-live");
  const resetButton = document.querySelector("#reset-scene");
  const outcome = document.querySelector("#sim-outcome");
  const outcomeKicker = document.querySelector("#outcome-kicker");
  const outcomeTitle = document.querySelector("#outcome-title");
  const outcomeReason = document.querySelector("#outcome-reason");
  const outcomeRestart = document.querySelector("#outcome-restart");
  const scenarioLabel = document.querySelector("#sim-scenario-label");
  const signImage = document.querySelector("#active-sign-image");
  const signTitle = document.querySelector("#active-sign-title");
  const signRule = document.querySelector("#active-sign-rule");
  const scenarioButtons = [...document.querySelectorAll("[data-scenario]")];
  const controlButtons = [...document.querySelectorAll("[data-control]")];

  const SCENARIOS = {
    yield: {
      title: "Yield", code: "2.4", image: "2.4.png",
      rule: "Give priority to vehicles on the priority road. Stop only when traffic makes it necessary.",
      prompt: "Read the traffic gap, yield if needed, then follow the green route.",
    },
    crosswalk: {
      title: "Pedestrian crossing", code: "5.19", image: "5.19.png",
      rule: "Yield to pedestrians entering or occupying the crossing, and do not stop on its approach zone.",
      prompt: "Watch the crossing and give pedestrians priority.",
    },
    detour: {
      title: "Detour", code: "4.2.1", image: "4.2.1.png",
      rule: "A lane obstacle blocks the route. Move right and pass it on the prescribed side.",
      prompt: "Change lane and pass the obstacle on the right.",
    },
    speed_limit: {
      title: "Maximum speed", code: "3.24", image: "3.24.png",
      rule: "Enter the signed zone at or below 20 km/h and stay within the posted maximum.",
      prompt: "Brake before the sign and hold 20 km/h through the zone.",
    },
  };

  const controls = {left: false, right: false, up: false, down: false};
  const pointers = new Map();
  let currentScenario = "yield";
  let socket = null;
  let connecting = false;
  let started = false;
  let renderingFrame = false;
  let queuedFrame = null;

  function resolveDemoBackendBase() {
    // Priority: ?ws=… → localStorage → <meta> → data-ws-url → same-origin.
    // GitHub Pages cannot host MetaDrive; pass a public wss:// tunnel, e.g.
    // https://emb-ai.github.io/traffic-sign-bench/?ws=wss://….trycloudflare.com/api/drive
    const params = new URLSearchParams(location.search);
    const fromQuery = params.get("ws")?.trim();
    if (fromQuery) {
      try {
        localStorage.setItem("traffic-bench-demo-ws", fromQuery);
      } catch (_) {
        /* ignore quota / private mode */
      }
      return fromQuery;
    }
    try {
      const stored = localStorage.getItem("traffic-bench-demo-ws")?.trim();
      if (stored) return stored;
    } catch (_) {
      /* ignore */
    }
    return (
      document.querySelector('meta[name="traffic-bench-demo-ws"]')?.content.trim() ||
      section?.dataset.wsUrl?.trim() ||
      ""
    );
  }

  function websocketUrl() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    let configured = resolveDemoBackendBase();
    if (configured && !/\/api\/drive\/?$/i.test(configured.replace(/\/$/, ""))) {
      configured = configured.replace(/\/$/, "") + "/api/drive";
    }
    const base = configured || `${protocol}//${location.host}/api/drive`;
    const url = new URL(base, location.href);
    if (url.protocol === "http:") url.protocol = "ws:";
    if (url.protocol === "https:") url.protocol = "wss:";
    url.pathname = `${url.pathname.replace(/\/$/, "")}/${encodeURIComponent(currentScenario)}`;
    url.searchParams.set("scenario", currentScenario);
    return url.toString();
  }

  function isVisible() {
    const rect = canvas.getBoundingClientRect();
    return rect.bottom > 0 && rect.top < window.innerHeight;
  }

  function setConnection(label, state) {
    if (connectionLabel) connectionLabel.textContent = label;
    section?.setAttribute("data-connection", state);
  }

  function showOverlay(title, copy) {
    if (startTitle) startTitle.textContent = title;
    if (startCopy) startCopy.textContent = copy;
    startOverlay?.classList.remove("is-hidden");
  }

  function hideOverlay() {
    startOverlay?.classList.add("is-hidden");
  }

  function hideOutcome() {
    if (outcome) outcome.hidden = true;
  }

  function showOutcome(data) {
    if (!outcome) return;
    const passed =
      data.result === "passed" ||
      (data.finished && data.state === "safe");
    outcome.dataset.result = passed ? "passed" : "failed";
    outcomeKicker.textContent = passed ? "Destination reached" : "Game over · rule violation";
    outcomeTitle.textContent = passed ? "Challenge complete" : `${SCENARIOS[currentScenario].title} violation`;
    outcomeReason.textContent = passed
      ? "You obeyed the sign and reached the destination."
      : data.message || "The executable checker detected a violation.";
    outcome.hidden = false;
  }

  function paintPlaceholder() {
    ctx.fillStyle = "#20272c";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(255, 255, 255, .72)";
    ctx.font = "600 15px DM Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Real MetaDrive frames will appear here", canvas.width / 2, canvas.height / 2);
  }

  function applyScenario(scenarioId) {
    const scenario = SCENARIOS[scenarioId];
    if (!scenario) return;

    releaseControls();
    currentScenario = scenarioId;
    started = false;
    connecting = false;
    hideOutcome();
    if (socket) {
      socket.demoIntentionalClose = true;
      socket.close(1000, "scenario changed");
    }
    socket = null;
    queuedFrame = null;

    scenarioButtons.forEach((button) => {
      const active = button.dataset.scenario === scenarioId;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    if (scenarioLabel) {
      scenarioLabel.textContent = `Moscow challenge · sign ${scenario.code}`;
    }
    if (signImage) {
      signImage.src = `static/images/signs/${scenario.image}`;
      signImage.alt = `${scenario.title} sign`;
    }
    if (signTitle) signTitle.textContent = `${scenario.title} · ${scenario.code}`;
    if (signRule) signRule.textContent = scenario.rule;
    if (startCopy) startCopy.textContent = scenario.prompt;
    if (status) status.textContent = "Start MetaDrive to run the checker";
    if (speedOutput) speedOutput.textContent = "0.0";
    if (distanceOutput) distanceOutput.textContent = "—";
    if (stopOutput) stopOutput.textContent = "Not started";
    if (verifier) verifier.dataset.state = "ready";
    setConnection("Ready to connect", "ready");
    showOverlay("Start simulation", scenario.prompt);
    paintPlaceholder();
  }

  function setControl(control, pressed) {
    if (controls[control] === pressed) return;
    controls[control] = pressed;
    const button = controlButtons.find((item) => item.dataset.control === control);
    button?.classList.toggle("is-pressed", pressed);
    sendControls();
  }

  function releaseControls() {
    Object.keys(controls).forEach((control) => setControl(control, false));
  }

  function send(payload) {
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }

  function sendControls() {
    send({type: "control", ...controls});
  }

  function updateTelemetry(data) {
    const speed = Number(data.speed_mps);
    speedOutput.textContent = Number.isFinite(speed) ? speed.toFixed(1) : "—";
    distanceOutput.textContent = Number.isFinite(data.distance_to_line_m)
      ? `${data.distance_to_line_m.toFixed(1)} m`
      : "—";
    stopOutput.textContent = data.result === "passed"
      ? "Passed ✓"
      : data.result === "failed"
        ? "Failed"
        : data.stopped
          ? "Stop registered ✓"
          : "In progress";
    status.textContent = data.message || "Checker is running";
    verifier.dataset.state = data.state || "ready";
    liveRegion.textContent = data.message || "";
    if (data.finished) {
      releaseControls();
      showOutcome(data);
    }
  }

  async function drawFrame(blob) {
    if (renderingFrame) {
      queuedFrame = blob;
      return;
    }
    renderingFrame = true;
    try {
      const bitmap = await createImageBitmap(blob);
      const sourceSize = Math.min(bitmap.width, bitmap.height);
      const sourceX = (bitmap.width - sourceSize) / 2;
      const sourceY = (bitmap.height - sourceSize) / 2;
      ctx.drawImage(
        bitmap,
        sourceX,
        sourceY,
        sourceSize,
        sourceSize,
        0,
        0,
        canvas.width,
        canvas.height,
      );
      bitmap.close();
    } finally {
      renderingFrame = false;
      if (queuedFrame) {
        const next = queuedFrame;
        queuedFrame = null;
        drawFrame(next);
      }
    }
  }

  function connect() {
    if (
      connecting ||
      socket?.readyState === WebSocket.OPEN ||
      socket?.readyState === WebSocket.CONNECTING
    ) return;

    connecting = true;
    started = true;
    setConnection("Connecting to MetaDrive…", "connecting");
    status.textContent = "Starting the real simulator";
    hideOverlay();

    try {
      socket = new WebSocket(websocketUrl());
      socket.binaryType = "blob";
    } catch (error) {
      connecting = false;
      setConnection("Demo unavailable", "offline");
      showOverlay(
        "Sorry, the live demo is unavailable",
        "We are trying to fix it. Please try again later.",
      );
      status.textContent =
        "Sorry, the live demo is unavailable. We are trying to fix it.";
      return;
    }

    const activeSocket = socket;
    activeSocket.addEventListener("open", () => {
      connecting = false;
      setConnection("Live MetaDrive", "live");
      sendControls();
      canvas.focus({preventScroll: true});
    });

    activeSocket.addEventListener("message", (event) => {
      if (event.data instanceof Blob) {
        drawFrame(event.data);
        return;
      }
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (_error) {
        return;
      }
      if (data.type === "telemetry") {
        updateTelemetry(data);
      } else if (data.type === "ready") {
        if (Number.isFinite(data.active_sessions) && Number.isFinite(data.max_sessions)) {
          setConnection(
            `Live MetaDrive · ${data.active_sessions}/${data.max_sessions} sessions`,
            "live",
          );
        }
        // Only enforce when the backend actually reports a scenario id.
        // Older single-scene servers omit it; closing then looked like a disconnect.
        if (data.scenario_id && data.scenario_id !== currentScenario) {
          status.textContent =
            `Backend opened ${data.scenario_id}; expected ${currentScenario}. Restart the demo server.`;
          verifier.dataset.state = "violation";
          activeSocket.demoIntentionalClose = true;
          activeSocket.close(1008, "scenario mismatch");
          setConnection("Scenario mismatch", "offline");
          showOverlay(
            "Sorry, the live demo is unavailable",
            "We are trying to fix it. Please try again later.",
          );
        }
      } else if (data.type === "error") {
        activeSocket.demoServerError = true;
        status.textContent = data.message;
        verifier.dataset.state = "violation";
        const atCapacity = data.code === "capacity";
        const timedOut = data.code === "idle_timeout";
        setConnection(atCapacity ? "All demo slots are busy" : "Simulator error", "offline");
        showOverlay(
          atCapacity
            ? "Demo at capacity"
            : timedOut
              ? "Session time reached"
              : "Could not start scene",
          data.message || "Click to retry",
        );
      }
    });

    activeSocket.addEventListener("close", () => {
      if (activeSocket.demoIntentionalClose || socket !== activeSocket) return;
      connecting = false;
      releaseControls();
      socket = null;
      if (activeSocket.demoServerError) return;
      setConnection("Demo unavailable", "offline");
      showOverlay(
        "Sorry, the live demo is unavailable",
        "We are trying to fix it. Please try again later.",
      );
    });

    activeSocket.addEventListener("error", () => {
      if (socket !== activeSocket || activeSocket.demoIntentionalClose) return;
      status.textContent =
        "Sorry, the live demo is unavailable. We are trying to fix it.";
      verifier.dataset.state = "violation";
      setConnection("Demo unavailable", "offline");
      showOverlay(
        "Sorry, the live demo is unavailable",
        "We are trying to fix it. Please try again later.",
      );
    });
  }

  const keyControls = {
    ArrowLeft: "left",
    ArrowRight: "right",
    ArrowUp: "up",
    ArrowDown: "down",
  };

  window.addEventListener("keydown", (event) => {
    const control = keyControls[event.key];
    if (!control || !isVisible()) return;
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
    event.preventDefault();
    connect();
    setControl(control, true);
  });

  window.addEventListener("keyup", (event) => {
    const control = keyControls[event.key];
    if (!control) return;
    event.preventDefault();
    setControl(control, false);
  });

  window.addEventListener("blur", releaseControls);

  controlButtons.forEach((button) => {
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      connect();
      const control = button.dataset.control;
      pointers.set(event.pointerId, control);
      button.setPointerCapture(event.pointerId);
      setControl(control, true);
    });

    const release = (event) => {
      const control = pointers.get(event.pointerId);
      if (!control) return;
      pointers.delete(event.pointerId);
      setControl(control, [...pointers.values()].includes(control));
    };

    button.addEventListener("pointerup", release);
    button.addEventListener("pointercancel", release);
    button.addEventListener("lostpointercapture", release);
  });

  startOverlay?.addEventListener("click", connect);
  canvas.addEventListener("pointerdown", connect);
  resetButton?.addEventListener("click", () => {
    releaseControls();
    hideOutcome();
    if (socket?.readyState === WebSocket.OPEN) {
      send({type: "restart"});
      status.textContent = `Restarting the ${SCENARIOS[currentScenario].title.toLowerCase()} scene…`;
      verifier.dataset.state = "ready";
    } else {
      connect();
    }
  });
  outcomeRestart?.addEventListener("click", () => {
    hideOutcome();
    releaseControls();
    if (socket?.readyState === WebSocket.OPEN) {
      send({type: "restart"});
      status.textContent = "Restarting scenario…";
      verifier.dataset.state = "ready";
      canvas.focus({preventScroll: true});
    } else {
      connect();
    }
  });

  scenarioButtons.forEach((button) => {
    button.addEventListener("click", () => applyScenario(button.dataset.scenario));
  });

  setInterval(() => {
    if (started && Object.values(controls).some(Boolean)) sendControls();
  }, 250);

  applyScenario("yield");
})();
